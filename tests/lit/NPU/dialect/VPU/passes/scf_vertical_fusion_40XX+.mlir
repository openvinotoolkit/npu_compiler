//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --scf-vertical-fusion --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 26)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>

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
    %1 = VPU.NCE.Convolution(%0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_9) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 26 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
    //CHECK:        [[OUT_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:        [[PAD_LOW5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP4]]([[OUT_SIZE]], [[TEMP_VALUE0]])
    //CHECK:        [[PAD_HIGH5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE2]]]
    //CHECK:        [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE0]])
    //CHECK:        [[TEMP_VALUE4:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE0]])
    //CHECK:        [[PAD_LOW4:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:        [[IN_SIZE0:%.+]] = affine.max #[[$MAP5]]([[TEMP_VALUE3]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH4:%.+]] = affine.min #[[$MAP3]]()[[[IN_SIZE0]]]
    //CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE3]])
    //CHECK:        [[TEMP_VALUE6:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE3]])
    //CHECK:        [[PAD_LOW3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE6]]]
    //CHECK:        [[TEMP_VALUE7:%.+]] = affine.max #[[$MAP6]]([[TEMP_VALUE5]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE7]]]
    //CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE5]])
    //CHECK:        [[TEMP_VALUE9:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE5]])
    //CHECK:        [[PAD_LOW2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE9]]]
    //CHECK:        [[TEMP_VALUE10:%.+]] = affine.max #[[$MAP7]]([[TEMP_VALUE8]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE10]]]
    //CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE8]])
    //CHECK:        [[TEMP_VALUE12:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE8]])
    //CHECK:        [[PAD_LOW1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE12]]]
    //CHECK:        [[TEMP_VALUE13:%.+]] = affine.max #[[$MAP8]]([[TEMP_VALUE11]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE13]]]
    //CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE11]])
    //CHECK:        [[TEMP_VALUE14:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE11]])
    //CHECK:        [[PAD_LOW0:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE14]]]
    //CHECK:        [[TEMP_VALUE15:%.+]] = affine.max #[[$MAP9]]([[SLICE_OFFSET]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH0:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE15]]]
    //CHECK:        [[SIZE_H:%.+]] = affine.apply #map10([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SIZE_H]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    //CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 44)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>
//CHECK: #[[$MAP10:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 + d10 - d11 - d12 + 12)>

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
    %1 = VPU.NCE.Convolution(%0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_6) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 44 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    //CHECK: [[CAST_INPUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs([[INPUT]] : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
    //CHECK:        [[OUTPUT_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER]])
    //CHECK:        [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER]])
    //CHECK:        [[PAD_LOW5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:        [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP4]]([[OUTPUT_SIZE]], [[TEMP_VALUE0]])
    //CHECK:        [[PAD_HIGH5:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE2]]]
    //CHECK:        [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE0]])
    //CHECK:        [[TEMP_VALUE4:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE0]])
    //CHECK:        [[PAD_LOW4:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:        [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP5]]([[TEMP_VALUE3]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH4:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE5]]]
    //CHECK:        [[TEMP_VALUE6:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE3]])
    //CHECK:        [[TEMP_VALUE7:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE3]])
    //CHECK:        [[PAD_LOW3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE7]]]
    //CHECK:        [[TEMP_VALUE8:%.+]] = affine.max #[[$MAP6]]([[TEMP_VALUE6]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH3:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE8]]]
    //CHECK:        [[TEMP_VALUE9:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE6]])
    //CHECK:        [[TEMP_VALUE10:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE6]])
    //CHECK:        [[PAD_LOW2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE10]]]
    //CHECK:        [[TEMP_VALUE11:%.+]] = affine.max #[[$MAP7]]([[TEMP_VALUE9]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE11]]]
    //CHECK:        [[TEMP_VALUE12:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE9]])
    //CHECK:        [[TEMP_VALUE13:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE9]])
    //CHECK:        [[PAD_LOW1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE13]]]
    //CHECK:        [[TEMP_VALUE14:%.+]] = affine.max #[[$MAP8]]([[TEMP_VALUE12]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE14]]]
    //CHECK:        [[SLICE_OFFSET:%.+]] = affine.max #[[$MAP1]]([[TEMP_VALUE12]])
    //CHECK:        [[TEMP_VALUE15:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE12]])
    //CHECK:        [[PAD_LOW0:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE15]]]
    //CHECK:        [[SLICE_SIZE:%.+]] = affine.max #[[$MAP9]]([[SLICE_OFFSET]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[PAD_HIGH0:%.+]] = affine.min #[[$MAP3]]()[[[SLICE_SIZE]]]
    //CHECK:        [[SLICE_SIZE:%.+]] = affine.apply #map10([[PAD_LOW0]], [[PAD_HIGH0]], [[PAD_LOW1]], [[PAD_HIGH1]], [[PAD_LOW2]], [[PAD_HIGH2]], [[PAD_LOW3]], [[PAD_HIGH3]], [[PAD_LOW4]], [[PAD_HIGH4]], [[OUTPUT_SIZE]], [[PAD_LOW5]], [[PAD_HIGH5]])
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SLICE_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUTPUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    scf.yield [[INSERT]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:    [[CAST:%.+]] = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs([[LOOP]]
    //CHECK:    return [[CAST]] : tensor<1x128x540x240xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>

// CHECK-LABEL: @MergeDynamicEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
func.func @MergeDynamicEltwise(
         %arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
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

     %1 = VPU.NCE.Eltwise(%0, %0) {
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

    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 240 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index

    //CHECK: [[DIM:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM]]) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]]  = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[SLICE_SIZE:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    //CHECK:                [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:  return [[LOOP]] :  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (-d0 + 960, 13)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0) -> (-d0 + 960, 27)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>

// CHECK-LABEL: @MergeVF2Chains
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x540x960xf16, {order = #NHWC}>)
func.func @MergeVF2Chains(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.Sign(%1) : tensor<1x32x540x960xf16, {order = #NHWC}>  -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.Convolution(%3, %cst_1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %4: tensor<1x32x540x960xf16, {order = #NHWC}>

    //CHECK: [[LOOP_STEP1:%.+]] = arith.constant 27 : index
    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP0:%.+]] = arith.constant 13 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 960 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_OUTPUT0:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP0:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER0:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP0]]
    //CHECK-SAME:           iter_args([[LOOP_OUT0:%arg[0-9]]] = [[LOOP_OUTPUT0]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    //CHECK:                [[INSERT_SIZE0:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER0]])
    //CHECK:                [[SLICE_OFFSET0:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER0]])
    //CHECK:                [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER0]])
    //CHECK:                [[PAD_LOW0:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE0]]]
    //CHECK:                [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP4]]([[INSERT_SIZE0]], [[SLICE_OFFSET0]])
    //CHECK:                [[PAD_HIGH0:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE1]]]
    //CHECK:                [[SLICE_SIZE0:%.+]] = affine.apply #[[$MAP5]]([[INSERT_SIZE0]], [[PAD_LOW0]], [[PAD_HIGH0]])

    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[SLICE_OFFSET0]]] [1, 32, 540, [[SLICE_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>
    //CHECK:                [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]]

    //CHECK:                [[PAD0:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]

    //CHECK:                [[INSERT0:%.+]] = tensor.insert_slice [[CONV0]] into [[LOOP_OUT0]][0, 0, 0, [[LOOP_ITER0]]] [1, 32, 540, [[INSERT_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT0]] : tensor<1x32x540x960xf16, {order = #NHWC}>

    //CHECK: [[SIGN:%.+]] = VPU.Sign([[LOOP0]]) : tensor<1x32x540x960xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT1:%.+]] = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: [[LOOP1:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER1:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP1]]
    //CHECK-SAME:           iter_args([[LOOP_OUT1:%arg[0-9]]] = [[LOOP_OUTPUT1]]) -> (tensor<1x32x540x960xf16, {order = #NHWC}>)

    //CHECK:                [[INSERT_SIZE1:%.+]] = affine.min #[[$MAP6]]([[LOOP_ITER1]])
    //CHECK:                [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP1]]([[LOOP_ITER0]])
    //CHECK:                [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER0]])
    //CHECK:                [[PAD_LOW2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE3]]]
    //CHECK:                [[TEMP_VALUE4:%.+]] = affine.max #[[$MAP4]]([[INSERT_SIZE1]], [[TEMP_VALUE2]])
    //CHECK:                [[PAD_HIGH2:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:                [[SLICE_OFFSET1:%.+]]  = affine.max #[[$MAP1]]([[TEMP_VALUE2]])
    //CHECK:                [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP2]]([[TEMP_VALUE2]])
    //CHECK:                [[PAD_LOW1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE5]]]
    //CHECK:                [[TEMP_VALUE6:%.+]] = affine.max #[[$MAP7]]([[SLICE_OFFSET1]], [[INSERT_SIZE1]], [[PAD_LOW2]], [[PAD_HIGH2]])
    //CHECK:                [[PAD_HIGH1:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE6]]]
    //CHECK:                [[SLICE_SIZE1:%.+]] = affine.apply #[[$MAP8]]([[PAD_LOW1]], [[PAD_HIGH1]], [[INSERT_SIZE1]], [[PAD_LOW2]], [[PAD_HIGH2]])

    //CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[SIGN]][0, 0, 0, [[SLICE_OFFSET1]]] [1, 32, 540, [[SLICE_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {order = #NHWC}>

    //CHECK:                [[PAD1:%.+]] = tensor.pad [[SLICE1]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]]
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]

    //CHECK:                [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {order = #NHWC}> to tensor<1x32x542x?xf16, {order = #NHWC}>
    //CHECK:                [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]

    //CHECK:                [[INSERT1:%.+]] = tensor.insert_slice [[CONV2]] into [[LOOP_OUT1]][0, 0, 0, [[LOOP_ITER1]]] [1, 32, 540, [[INSERT_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
    //CHECK: return [[LOOP1]] : tensor<1x32x540x960xf16, {order = #NHWC}>
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
     %2 = VPU.NCE.Eltwise(%0, %1) {
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


    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 70 : index
    //CHECK: [[LOOP_END:%.+]] = arith.constant 140 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty() : tensor<1x16x256x140xf16, {order = #NHWC}>

    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x256x140xf16, {order = #NHWC}>)

    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT0]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    //CHECK:                [[CAST0:%.+]] = VPU.LayoutCast([[SLICE0]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    //CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[INPUT1]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x140xf16> to tensor<1x16x256x70xf16>
    //CHECK:                [[CAST1:%.+]] = VPU.LayoutCast([[SLICE1]]) {dst_order = #NHWC} : tensor<1x16x256x70xf16> -> tensor<1x16x256x70xf16, {order = #NHWC}>
    //CHECK:                [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CAST0]], [[CAST1]])
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, 70] [1, 1, 1, 1] : tensor<1x16x256x70xf16, {order = #NHWC}> into tensor<1x16x256x140xf16, {order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]] : tensor<1x16x256x140xf16, {order = #NHWC}>

    // pure view-like op doesn't have tilingStrategy, it cannot be tiled. it might be used to continue VF further, but we cannot start VF with that unfortunately
    //CHECK: [[CAST2:%.+]] = VPU.LayoutCast([[LOOP]]) {dst_order = #NCHW} : tensor<1x16x256x140xf16, {order = #NHWC}> -> tensor<1x16x256x140xf16>
    //CHECK: return [[CAST2]] : tensor<1x16x256x140xf16>
}

// -----

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 32)>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

 // CHECK-LABEL: @MergeConvertPermute
 // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
 func.func @MergeConvertPermute(
         %arg0: tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
 ) -> tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}> {

    %0 = VPU.Convert(%arg0) {dstElemType = f16,
                            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                            tilingStrategy = [1, 1, 1, 26]}
                            : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
                            -> tensor<1x3x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>

    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
                            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                            tilingStrategy = [1, 1, 1, 80]}
                            -> tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    return %1 : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>


    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 32 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM0:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM0]]) : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[INSERT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 3, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 32]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]])
    //CHECK:                [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 32]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                scf.yield [[INSERT]]

    //CHECK: return %1 : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 120)>

// CHECK-LABEL: @Merge2DDynamicEltwise
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
func.func @Merge2DDynamicEltwise(
         %arg0: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
     %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
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

     %1 = VPU.NCE.Eltwise(%0, %0) {
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

    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 120 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 16 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]

    //CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 120]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    //CHECK:                  [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    //CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 16, 120]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  scf.yield [[INSERT]]
    //CHECK:  scf.yield [[LOOP_W]]

    //CHECK: return [[LOOP_H]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 45)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 240)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP4:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 538)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 536)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (-d0 - d1 + d2 - d3 - d4 + 4)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>

// CHECK-LABEL: @Merge2DVFChain3Tiles
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>)
func.func @Merge2DVFChain3Tiles(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                           pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                           ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                           lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                           rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 21]}
        : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %1 = VPU.NCE.DepthConvolution(%0, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                                       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                                       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                                       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                                       rawFilterShape = [32, 1, 1, 1], strides = [1, 1], tilingStrategy = [1, 1, 1, 20]}
        -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    %2 = VPU.NCE.Convolution(%1, %cst_1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                                          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                                          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                                          ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                                          rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 1, 22]}
        : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>,
          tensor<32x32x3x3xf16, {order = #NHWC}>
          -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    return %2: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 45 : index
    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[INSERT_SIZE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                  [[INSERT_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]
    //CHECK:                  [[TMP_VALUE7:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_H]])
    //CHECK:                  [[TMP_VALUE6:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_H]])
    //CHECK:                  [[PAD1_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE6]]]
    //CHECK:                  [[TMP_VALUE9:%.+]] = affine.max #[[$MAP5]]([[INSERT_SIZE_H]], [[TMP_VALUE7]])
    //CHECK:                  [[PAD1_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE9]]]
    //CHECK:                  [[TMP_VALUE5:%.+]] = affine.max #[[$MAP2]]([[LOOP_ITER_W]])
    //CHECK:                  [[TMP_VALUE8:%.+]] = affine.max #[[$MAP3]]([[LOOP_ITER_W]])
    //CHECK:                  [[PAD1_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE8]]]
    //CHECK:                  [[TMP_VALUE4:%.+]] = affine.max #[[$MAP6]]([[INSERT_SIZE_W]], [[TMP_VALUE5]])
    //CHECK:                  [[PAD1_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE4]]]
    //CHECK:                  [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE7]])
    //CHECK:                  [[TMP_VALUE3:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE7]])
    //CHECK:                  [[PAD0_LOW_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE3]]]
    //CHECK:                  [[TMP_VALUE2:%.+]] = affine.max #[[$MAP7]]([[SLICE_OFFSET_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])
    //CHECK:                  [[PAD0_HIGH_H:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE2]]]
    //CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP8]]([[PAD0_LOW_H]], [[PAD0_HIGH_H]], [[INSERT_SIZE_H]], [[PAD1_LOW_H]], [[PAD1_HIGH_H]])
    //CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[TMP_VALUE5]])
    //CHECK:                  [[TMP_VALUE1:%.+]] = affine.max #[[$MAP3]]([[TMP_VALUE5]])
    //CHECK:                  [[PAD0_LOW_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE1]]]
    //CHECK:                  [[TMP_VALUE0:%.+]] = affine.max #[[$MAP9]]([[SLICE_OFFSET_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])
    //CHECK:                  [[PAD0_HIGH_W:%.+]] = affine.min #[[$MAP4]]()[[[TMP_VALUE0]]]
    //CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP8]]([[PAD0_LOW_W]], [[PAD0_HIGH_W]], [[INSERT_SIZE_W]], [[PAD1_LOW_W]], [[PAD1_HIGH_W]])

    //CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 32, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD0_LOW_H]], [[PAD0_LOW_W]]] high[0, 0, [[PAD0_HIGH_H]], [[PAD0_HIGH_W]]] {
    //CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  [[CONV0:%.+]]  = VPU.NCE.Convolution([[PAD0]]
    //CHECK:                  [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    //CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    //CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[CONV1]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[INSERT_SIZE_H]], [[INSERT_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  scf.yield [[INSERT]]

    //CHECK:  scf.yield [[LOOP_W]]

    //CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
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

    // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant {{[0-9]+}} : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    // CHECK:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
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
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (0, d0 - 1)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 48)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 798)>
//CHECK: #[[$MAP3:.*]] = affine_map<()[s0] -> (1, s0)>
//CHECK: #[[$MAP4:.*]] = affine_map<(d0) -> (-d0 + 1, 0)>
//CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 796)>
//CHECK: #[[$MAP6:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 320)>
//CHECK: #[[$MAP7:.*]] = affine_map<(d0, d1) -> (0, d0 + d1 - 1278)>
//CHECK: #[[$MAP8:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 1276)>
//CHECK: #[[$MAP9:.*]] = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>

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
    
    %0 = VPU.NCE.Convolution(%arg0, %cst) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, 
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]
    }  : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> 
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 
     
    %1 = VPU.NCE.DepthConvolution(%0, %cst_1) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [32, 1, 1, 1], strides = [1, 1]
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 
    
    %2 = VPU.NCE.Convolution(%1, %cst_3) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, 
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> 
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 
    
    %3 = VPU.NCE.Convolution(%2, %cst_5) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, 
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, 
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]
    } : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>
      -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 
    
    %4 = VPU.NCE.DepthConvolution(%3, %cst_7) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        tilingStrategy = [1, 1, 1, 20],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, 
        rawFilterShape = [32, 1, 1, 1], strides = [1, 1]
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 
    
    %5 = VPU.NCE.Eltwise(%4, %1) {
        is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        op_type = #VPU.eltwise_type<ADD>, tilingStrategy = [1, 1, 1, 20],
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, 
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], 
        fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> 

    return %5 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>   
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 320 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 48 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index

    //CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H]], [[DIM_W]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK: [[LOOP_H:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    //CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[LOOP_W:%.+]] = scf.for
    //CHECK-SAME:             [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
    //CHECK-SAME:             iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                  [[TEMP_VALUE0:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                  [[TEMP_VALUE1:%.+]] = affine.max #[[$MAP0]]([[TEMP_VALUE0]])
    //CHECK:                  [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    //CHECK:                  [[TEMP_VALUE2:%.+]] = affine.max #[[$MAP2]]([[SLICE_SIZE_H]], [[TEMP_VALUE0]])
    //CHECK:                  [[PAD2_HIGH_H:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE2]]]
    //CHECK:                  [[TEMP_VALUE3:%.+]] = affine.max #[[$MAP4]]([[LOOP_ITER_H]])
    //CHECK:                  [[PAD2_LOW_H:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE3]]]
    //CHECK:                  [[TEMP_VALUE4:%.+]] = affine.max #[[$MAP5]]([[TEMP_VALUE1]], [[SLICE_SIZE_H]], [[PAD2_LOW_H]], [[PAD2_HIGH_H]])
    //CHECK:                  [[PAD1_HIGH_H:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE4]]]
    //CHECK:                  [[TEMP_VALUE5:%.+]] = affine.max #[[$MAP4]]([[TEMP_VALUE0]])
    //CHECK:                  [[PAD1_LOW_H:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE5]]]
    //CHECK:                  [[TEMP_VALUE6:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_W]])
    //CHECK:                  [[TEMP_VALUE7:%.+]] = affine.max #[[$MAP0]]([[TEMP_VALUE6]])
    //CHECK:                  [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP6]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]
    //CHECK:                  [[TEMP_VALUE8:%.+]] = affine.max #[[$MAP7]]([[SLICE_SIZE_W]], [[TEMP_VALUE6]])
    //CHECK:                  [[PAD2_HIGH_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE8]]]
    //CHECK:                  [[TEMP_VALUE9:%.+]] = affine.max #[[$MAP4]]([[LOOP_ITER_W]])
    //CHECK:                  [[PAD2_LOW_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE9]]]
    //CHECK:                  [[TEMP_VALUE10:%.+]] = affine.max #[[$MAP8]]([[TEMP_VALUE7]], [[SLICE_SIZE_W]], [[PAD2_LOW_W]], [[PAD2_HIGH_W]])
    //CHECK:                  [[PAD1_HIGH_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE10]]]
    //CHECK:                  [[TEMP_VALUE11:%.+]] = affine.max #[[$MAP4]]([[TEMP_VALUE6]])
    //CHECK:                  [[PAD1_LOW_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE11]]]
    //CHECK:                  [[SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_W]])
    //CHECK:                  [[TEMP_VALUE13:%.+]] = affine.max #[[$MAP7]]([[SLICE_SIZE_W]], [[SLICE_OFFSET_W]])
    //CHECK:                  [[PAD0_HIGH_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE13]]]
    //CHECK:                  [[TEMP_VALUE14:%.+]] = affine.max #[[$MAP4]]([[LOOP_ITER_W]])
    //CHECK:                  [[PAD0_LOW_W:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE14]]]
    //CHECK:                  [[EXTRACT_SIZE_W:%.+]] = affine.apply #[[$MAP9]]([[SLICE_SIZE_W]], [[PAD0_LOW_W]], [[PAD0_HIGH_W]])
    //CHECK:                  [[SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP0]]([[LOOP_ITER_H]])
    //CHECK:                  [[TEMP_VALUE16:%.+]] = affine.max #[[$MAP2]]([[SLICE_SIZE_H]], [[SLICE_OFFSET_H]])
    //CHECK:                  [[PAD0_HIGH_H:%.+]]  = affine.min #[[$MAP3]]()[[[TEMP_VALUE16]]]
    //CHECK:                  [[TEMP_VALUE17:%.+]] = affine.max #[[$MAP4]]([[LOOP_ITER_H]])
    //CHECK:                  [[PAD0_LOW_H:%.+]] = affine.min #[[$MAP3]]()[[[TEMP_VALUE17]]]
    //CHECK:                  [[EXTRACT_SIZE_H:%.+]] = affine.apply #[[$MAP9]]([[SLICE_SIZE_H]], [[PAD0_LOW_H]], [[PAD0_HIGH_H]])

    //CHECK:                  [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 32, [[EXTRACT_SIZE_H]], [[EXTRACT_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD0_LOW_H]], [[PAD0_LOW_W]]] high[0, 0, [[PAD0_HIGH_H]], [[PAD0_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
    
    //CHECK:                  [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]] 
    //CHECK:                  [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    
    //CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]] 
    //CHECK:                  [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, [[PAD2_LOW_H]], [[PAD2_LOW_W]]] high[0, 0, [[PAD2_HIGH_H]], [[PAD2_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}>
    
    //CHECK:                  [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]] 
    //CHECK:                  [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV2]] 
    //CHECK:                  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[DWCONV1]], [[DWCONV0]])

    //CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 320]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  scf.yield [[INSERT]]
    //CHECK:  scf.yield [[LOOP_W]]

    //CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}
