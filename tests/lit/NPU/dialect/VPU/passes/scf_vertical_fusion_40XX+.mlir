//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --scf-vertical-fusion --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
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
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SIZE_H]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
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
    //CHECK:        [[SLICE:%.+]] = tensor.extract_slice [[CAST_INPUT]][0, 0, 0, [[SLICE_OFFSET]]] [1, 32, 540, [[SLICE_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:        tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:        [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD1:%.+]] = tensor.pad [[CONV0]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD3:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW3]]] high[0, 0, 1, [[PAD_HIGH3]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV3:%.+]] = VPU.NCE.Convolution([[PAD3]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD4:%.+]] = tensor.pad [[CONV3]] low[0, 0, 1, [[PAD_LOW4]]] high[0, 0, 1, [[PAD_HIGH4]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV4:%.+]] = VPU.NCE.Convolution([[PAD4]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[PAD5:%.+]] = tensor.pad [[CONV4]] low[0, 0, 1, [[PAD_LOW5]]] high[0, 0, 1, [[PAD_HIGH5]]] {
    //CHECK:          tensor.yield [[PAD_VALUE]] : f16
    //CHECK:          tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:        [[CONV5:%.+]] = VPU.NCE.Convolution([[PAD5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV5]]
    //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    //CHECK:        [[INSERT:%.+]] = tensor.insert_slice [[DWCONV1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 32, 540, [[OUTPUT_SIZE]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
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
//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 160)>

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

    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 160 : index
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
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 160]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[SLICE]], [[SLICE]])
    //CHECK:                [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]], [[ELTWISE0]])

    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE1]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 256, [[SLICE_SIZE]]] [1, 1, 1, 1]
    //CHECK-SAME:           tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 160]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
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

    //CHECK:                [[SLICE0:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[SLICE_OFFSET0]]] [1, 32, 540, [[SLICE_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]]

    //CHECK:                [[PAD0:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, 1, [[PAD_LOW0]]] high[0, 0, 1, [[PAD_HIGH0]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]

    //CHECK:                [[INSERT0:%.+]] = tensor.insert_slice [[CONV0]] into [[LOOP_OUT0]][0, 0, 0, [[LOOP_ITER0]]] [1, 32, 540, [[INSERT_SIZE0]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
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

    //CHECK:                [[SLICE1:%.+]] = tensor.extract_slice [[SIGN]][0, 0, 0, [[SLICE_OFFSET1]]] [1, 32, 540, [[SLICE_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                [[PAD1:%.+]] = tensor.pad [[SLICE1]] low[0, 0, 1, [[PAD_LOW1]]] high[0, 0, 1, [[PAD_HIGH1]]]
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]

    //CHECK:                [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, 1, [[PAD_LOW2]]] high[0, 0, 1, [[PAD_HIGH2]]] {
    //CHECK:                tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]

    //CHECK:                [[INSERT1:%.+]] = tensor.insert_slice [[CONV2]] into [[LOOP_OUT1]][0, 0, 0, [[LOOP_ITER1]]] [1, 32, 540, [[INSERT_SIZE1]]] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
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

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 64)>

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


    //CHECK: [[LOOP_STEP:%.+]] = arith.constant 64 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
    //CHECK: [[DIM_INDEX:%.+]] = arith.constant 3 : index
    //CHECK: [[DIM0:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM0]]) : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK: [[LOOP_END:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX]] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK: [[LOOP:%.+]] = scf.for
    //CHECK-SAME:           [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    //CHECK-SAME:           iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>) {

    //CHECK:                [[INSERT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
    //CHECK:                [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, 0, [[LOOP_ITER]]] [1, 3, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x3x1600x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 64]> : tensor<4xsi64>, order = #NCHW}>
    //CHECK:                [[CONVERT:%.+]] = VPU.Convert([[SLICE]])
    //CHECK:                [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
    //CHECK:                [[INSERT:%.+]] = tensor.insert_slice [[PERMUTE]] into [[LOOP_OUT]][0, 0, 0, [[LOOP_ITER]]] [1, 16, 1600, [[INSERT_SIZE]]] [1, 1, 1, 1] : tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 64]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x1600x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
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

    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 120 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 16 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
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

    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[PAD_VALUE:%.+]] = arith.constant 0.000000e+00 : f16
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 240 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 45 : index
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
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 242]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  [[CONV0:%.+]]  = VPU.NCE.Convolution([[PAD0]]
    //CHECK:                  [[DWCONV:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]
    //CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    //CHECK:                  tensor.yield [[PAD_VALUE]] : f16
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 242]> : tensor<4xsi64>, order = #NHWC}>
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
//CHECK: #[[$MAP6:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
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
    //CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    //CHECK: [[LOOP_STEP_W:%.+]] = arith.constant 256 : index
    //CHECK: [[LOOP_STEP_H:%.+]] = arith.constant 48 : index
    //CHECK: [[LOOP_BEGIN:%.+]] = arith.constant 0 : index
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
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[PAD0:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD0_LOW_H]], [[PAD0_LOW_W]]] high[0, 0, [[PAD0_HIGH_H]], [[PAD0_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 258]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[CONV0:%.+]] = VPU.NCE.Convolution([[PAD0]]
    //CHECK:                  [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[CONV0]]

    //CHECK:                  [[PAD1:%.+]] = tensor.pad [[DWCONV0]] low[0, 0, [[PAD1_LOW_H]], [[PAD1_LOW_W]]] high[0, 0, [[PAD1_HIGH_H]], [[PAD1_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 258]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  [[CONV1:%.+]] = VPU.NCE.Convolution([[PAD1]]
    //CHECK:                  [[PAD2:%.+]] = tensor.pad [[CONV1]] low[0, 0, [[PAD2_LOW_H]], [[PAD2_LOW_W]]] high[0, 0, [[PAD2_HIGH_H]], [[PAD2_HIGH_W]]] {
    //CHECK:                  tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 258]> : tensor<4xsi64>, order = #NHWC}>

    //CHECK:                  [[CONV2:%.+]] = VPU.NCE.Convolution([[PAD2]]
    //CHECK:                  [[DWCONV1:%.+]] = VPU.NCE.DepthConvolution([[CONV2]]
    //CHECK:                  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[DWCONV1]], [[DWCONV0]])

    //CHECK:                  [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 32, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1]
    //CHECK-SAME:             tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 48, 256]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    //CHECK:                  scf.yield [[INSERT]]
    //CHECK:  scf.yield [[LOOP_W]]

    //CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputF32DynamicType = tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputDynamicType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputConvDynamicType = tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputD2SDynamicType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!outputD2SDynamicType = tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>

module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 96)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 732)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0] -> (1, s0)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<(d0, d1) -> (d0 + d1 floordiv 2 - 1598, 0)>
// CHECK-DAG: #[[$MAP6:.+]] = affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 floordiv 2 + 2)>
// CHECK-DAG: #[[$MAP7:.+]] = affine_map<(d0, d1) -> (d0 + d1 floordiv 2 - 2558, 0)>

// CHECK: @Merge2DVFChainConvAddD2S
func.func @Merge2DVFChainConvAddD2S(%arg0: !inputConvDynamicType) -> !outputD2SDynamicType {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    %weights = const.Declare tensor<16x32x3x3xf16, {order = #NHWC}> = dense<1.0>
        : tensor<16x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                          lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [16, 32, 3, 3], strides = [1, 1],
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

    // CHECK:   [[DIM_H2:%.+]] = tensor.dim [[ARG0]], [[CST_VAL_2]]
    // CHECK:   [[OUT_DIM_H2:%.+]] = arith.muli [[DIM_H2]], [[CST_VAL_2]]

    // CHECK:   [[DIM_W2:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_W]]
    // CHECK:   [[OUT_DIM_W2:%.+]] = arith.muli [[DIM_W2]], [[CST_VAL_2]]

    // CHECK:   [[RESULT:%.+]] = scf.for [[SLICE_OFFSET_H:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_H2]] step [[LOOP_STEP_H]] iter_args([[OUTER_OUTPUT:%.+]] = [[EMPTY]])
    // CHECK:       [[RESULT_W:%.+]] = scf.for [[SLICE_OFFSET_W:%.+]] = [[LOOP_BEGIN]] to [[OUT_DIM_W2]] step [[LOOP_STEP_W]] iter_args([[INNER_OUTPUT:%.+]] = [[OUTER_OUTPUT]])

    // CHECK:       [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP]]([[SLICE_OFFSET_H]])[[[OUT_DIM_H2]]]
    // CHECK:       [[SLICE_SIZE_W:%.+]] = affine.min #[[$MAP1]]([[SLICE_OFFSET_W]])[[[OUT_DIM_W2]]]

    // CHECK:       [[IN_SLICE_OFFSET_H:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET_H]])
    // CHECK:       [[TEMP_VAL0:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_H]])
    // CHECK:       [[PAD_BOTTOM:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL0]]]
    // CHECK:       [[TEMP_VAL1:%.+]] = affine.max #[[$MAP5]]([[IN_SLICE_OFFSET_H]], [[SLICE_SIZE_H]])
    // CHECK:       [[PAD_TOP:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL1]]]
    // CHECK:       [[IN_SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP6]]([[PAD_BOTTOM]], [[PAD_TOP]], [[SLICE_SIZE_H]])

    // CHECK:       [[IN_SLICE_OFFSET_W:%.+]] = affine.max #[[$MAP2]]([[SLICE_OFFSET_W]])
    // CHECK:       [[TEMP_VAL2:%.+]] = affine.max #[[$MAP3]]([[SLICE_OFFSET_W]])
    // CHECK:       [[PAD_LEFT:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL2]]]
    // CHECK:       [[TEMP_VAL3:%.+]] = affine.max #[[$MAP7]]([[IN_SLICE_OFFSET_W]], [[SLICE_SIZE_W]])
    // CHECK:       [[PAD_RIGHT:%.+]] = affine.min #[[$MAP4]]()[[[TEMP_VAL3]]]
    // CHECK:       [[IN_SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP6]]([[PAD_LEFT]], [[PAD_RIGHT]], [[SLICE_SIZE_W]])

    // CHECK:       [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, 0, [[IN_SLICE_OFFSET_H]], [[IN_SLICE_OFFSET_W]]] [1, 32, [[IN_SLICE_SIZE_H]], [[IN_SLICE_SIZE_W]]] [1, 1, 1, 1]

    // CHECK:       [[PAD:%.+]] = tensor.pad [[SLICE]] low[0, 0, [[PAD_BOTTOM]], [[PAD_LEFT]]] high[0, 0, [[PAD_TOP]], [[PAD_RIGHT]]]

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[PAD]], [[WEIGHTS]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:    -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONV]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:    : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 366]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 96, 732]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[INSERT:%.+]] = tensor.insert_slice [[D2S]] into [[INNER_OUTPUT]]
    // CHECK:       scf.yield [[INSERT]]
    // CHECK:   scf.yield [[RESULT_W]]
    // CHECK:   return [[RESULT]] : tensor<1x4x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 4, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>
}
}

// -----

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 15)>

// CHECK-LABEL: @MergeVFPermuteCastAxis
func.func @MergeVFPermuteCastAxis(%arg0: tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>)
     -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, 
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1],
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
    // CHECK:   [[LOOP_END:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX]] : tensor<1x?x1920x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP:%.+]] = scf.for 
    // CHECK-SAME:             [[LOOP_ITER:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END]] step [[LOOP_STEP]]
    // CHECK-SAME:             iter_args([[LOOP_OUT:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>) {

    // CHECK:                 [[INSERT_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])[[[LOOP_END]]]
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

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
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
    %1 = VPU.NCE.Convolution(%0, %cst) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, 
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1],
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

config.Resources 6 of @NCE at 1.850000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 90)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 480)>

// CHECK-LABEL: @MergeVFPermuteCast2DAxis
func.func @MergeVFPermuteCast2DAxis(%arg0: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>)
     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    %cst = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, 
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, 
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1],
        tilingStrategy = [1, 1, 1, 80]}
        : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>, 
        tensor<16x16x1x1xf16, {order = #NHWC}>
     -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> 

    return %1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:   [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant 480 : index
    // CHECK:   [[LOOP_STEP_C:%.+]] = arith.constant 90 : index
    // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

    // CHECK:   [[DIM_INDEX_C:%.+]] = arith.constant 1 : index
    // CHECK:   [[DIM_C_0:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_C]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[DIM_H_0:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_H]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_C_0]], [[DIM_H_0]]) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    
    // CHECK:   [[LOOP_END_C:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_C]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP_END_H:%.+]] = tensor.dim [[ARG0]], [[DIM_INDEX_H]] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   [[LOOP_C:%.+]] = scf.for 
    // CHECK-SAME:              [[LOOP_ITER_C:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_C]] step [[LOOP_STEP_C]]
    // CHECK-SAME:              iter_args([[LOOP_OUT_C:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)
    // CHECK:                   [[LOOP_H:%.+]] = scf.for 
    // CHECK-SAME:              [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
    // CHECK-SAME:             iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUT_C]]) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>)
    
    // CHECK:                 [[SLICE_SIZE_C:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_C]])[[[LOOP_END_C]]]
    // CHECK:                 [[SLICE_SIZE_H:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
    // CHECK:                 [[SLICE:%.+]] = tensor.extract_slice [[ARG0]][0, [[LOOP_ITER_C]], [[LOOP_ITER_H]], 0] [1, [[SLICE_SIZE_C]], [[SLICE_SIZE_H]], 16] [1, 1, 1, 1] : tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 1080, 1920, 16]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x?x?x16xf16, {bounds = #const.OpaqueI64Elements<[1, 90, 480, 16]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:                 [[PERMUTECAST:%.+]] = VPU.PermuteCast([[SLICE]])
    // CHECK:                 [[CONV:%.+]] = VPU.NCE.Convolution([[PERMUTECAST]]
    // CHECK:                 [[INSERT:%.+]] = tensor.insert_slice [[CONV]] into [[LOOP_OUT_H]][0, 0, [[LOOP_ITER_C]], [[LOOP_ITER_H]]] [1, 16, [[SLICE_SIZE_C]], [[SLICE_SIZE_H]]] [1, 1, 1, 1] 
    // CHECK-SAME:            tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 480]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
      
    // CHECK:   scf.yield [[INSERT]]
    // CHECK:   scf.yield [[LOOP_H]]
    // CHECK:   return [[LOOP_C]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
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
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }

  // CHECK-LABEL: @MergeVFQuantizedChain2Tiles
  // CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
  func.func @MergeVFQuantizedChain2Tiles(%arg0: !inputDynamicType) 
      -> !outputDynamicType {
      %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x16x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, 0.034925088695451328>>, #const.ConvertElemType<!quant.uniform<ui8:f16, 0.034925088695451328:128>>, #const.Reorder<#NHWC>]
      %cst_1 = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, 0.0091306873396331187>>, #const.ConvertElemType<!quant.uniform<ui8:f16, 0.0091306873396331187:128>>, #const.Reorder<#NHWC>]    
      %3 = VPU.NCE.Convolution(%arg0, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.100000e+02 : f64, clamp_high = 1.450000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 1.100000e+02 : f64>, rawFilterShape = [32, 16, 3, 3], strides = [2, 2], tilingStrategy = [1, 1, 23, 4]} : !inputDynamicType, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x?x?x!quant.uniform<ui8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> 
      %4 = VPU.NCE.Convolution(%3, %cst_1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [-0.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 23, 4]} : tensor<1x32x?x?x!quant.uniform<ui8:f16, 0.030033121856988646:110>, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>     
      return %4: !outputDynamicType

    // CHECK: [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
    // CHECK: [[PAD_VALUE:%.+]] = arith.constant 110 : i8
    // CHECK: [[ZERO_PAD_VALUE:%.+]] = arith.constant 0 : i8
    // CHECK: [[TILE_STEP_W:%.+]] = arith.constant 240 : index
    // CHECK: [[TILE_STEP_H:%.+]] = arith.constant 45 : index
    // CHECK: [[START_INDEX:%.+]] = arith.constant 0 : index
    // CHECK: [[CONV1_WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x16x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>, #const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
    // CHECK: [[CONV2_WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1> : tensor<32x32x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType4>, #const.ConvertElemType<!qElemType3>, #const.Reorder<#NHWC>]
    // CHECK: [[DIM_INDEX_H:%.+]] = arith.constant 2 : index
    // CHECK: [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_HEIGHT:%.+]] = arith.divsi [[DIM_H]], [[DIM_INDEX_H]] : index
    // CHECK: [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_WIDTH:%.+]] = arith.divsi [[DIM_W]], [[DIM_INDEX_H]] : index
    // CHECK: [[OUTPUT_BUF:%.+]] = tensor.empty([[HALF_HEIGHT]], [[HALF_WIDTH]]) : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[DIM_H_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_H]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_HEIGHT_2:%.+]] = arith.divsi [[DIM_H_2]], [[DIM_INDEX_H]] : index
    // CHECK: [[DIM_W_2:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[HALF_WIDTH_2:%.+]] = arith.divsi [[DIM_W_2]], [[DIM_INDEX_H]] : index
    // CHECK: [[LOOP_H:%.+]] = scf.for [[LOOP_ITER:%arg[0-9]]] = [[START_INDEX:%c0]] to [[HALF_HEIGHT_2:%3]] step [[TILE_STEP_H:%c45]] iter_args([[LOOP_ITER_1:%arg[0-9]]] = [[OUTPUT_BUF:%2]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[LOOP_W:%.+]] = scf.for [[LOOP_ITER_2:%arg[0-9]]] = [[START_INDEX:%c0]] to [[HALF_WIDTH_2:%4]] step [[TILE_STEP_W:%c240]] iter_args([[LOOP_ITER_3:%arg[0-9]]] = [[LOOP_ITER_1]]) -> (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK: [[MIN_HEIGHT:%.+]] = affine.min #map([[LOOP_ITER]])[[[HALF_HEIGHT_2]]]
    // CHECK: [[MIN_WIDTH:%.+]] = affine.min #map1([[LOOP_ITER_2]])[[[HALF_WIDTH_2]]]
    // CHECK: [[MAX_HEIGHT_TEMP:%.+]] = affine.max #map2([[LOOP_ITER]])
    // CHECK: [[MAX_WIDTH_TEMP:%.+]] = affine.max #map3([[LOOP_ITER]])
    // CHECK: [[PAD_LOW_TEMP_H:%.+]] = affine.min #map4()[[[MAX_WIDTH_TEMP]]]
    // CHECK: [[MAX_HEIGHT_COMBINED:%.+]] = affine.max #map5([[MIN_HEIGHT]], [[MAX_HEIGHT_TEMP]])
    // CHECK: [[PAD_HIGH_TEMP_H:%.+]] = affine.min #map4()[[[MAX_HEIGHT_COMBINED]]]
    // CHECK: [[MAX_WIDTH_TEMP_2:%.+]] = affine.max #map2([[LOOP_ITER_2]])
    // CHECK: [[MAX_WIDTH_TEMP_3:%.+]] = affine.max #map3([[LOOP_ITER_2]])
    // CHECK: [[PAD_LOW_TEMP_W:%.+]] = affine.min #map4()[[[MAX_WIDTH_TEMP_3]]]
    // CHECK: [[MAX_WIDTH_COMBINED:%.+]] = affine.max #map6([[MIN_WIDTH]], [[MAX_WIDTH_TEMP_2]])
    // CHECK: [[PAD_HIGH_TEMP_W:%.+]] = affine.min #map4()[[[MAX_WIDTH_COMBINED]]]
    // CHECK: [[MAX_HEIGHT_FINAL:%.+]] = affine.max #map7([[MAX_HEIGHT_TEMP]])
    // CHECK: [[MAX_WIDTH_FINAL:%.+]] = affine.max #map8([[MAX_HEIGHT_TEMP]])
    // CHECK: [[PAD_LOW_FINAL_H:%.+]] = affine.min #map4()[[[MAX_WIDTH_FINAL]]]
    // CHECK: [[APPLY_HEIGHT:%.+]] = affine.apply #map9([[PAD_LOW_FINAL_H]], [[MIN_HEIGHT]], [[PAD_LOW_TEMP_H]], [[PAD_HIGH_TEMP_H]])
    // CHECK: [[MAX_WIDTH_FINAL_2:%.+]] = affine.max #map7([[MAX_WIDTH_TEMP_2]])
    // CHECK: [[MAX_WIDTH_FINAL_3:%.+]] = affine.max #map8([[MAX_WIDTH_TEMP_2]])
    // CHECK: [[PAD_LOW_FINAL_W:%.+]] = affine.min #map4()[[[MAX_WIDTH_FINAL_3]]]
    // CHECK: [[APPLY_WIDTH:%.+]] = affine.apply #map9([[PAD_LOW_FINAL_W]], [[MIN_WIDTH]], [[PAD_LOW_TEMP_W]], [[PAD_HIGH_TEMP_W]])
    // CHECK: [[EXTRACTED_SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[MAX_HEIGHT_FINAL]], [[MAX_WIDTH_FINAL_2]]] [1, 16, [[APPLY_HEIGHT]], [[APPLY_WIDTH]]] [1, 1, 1, 1] : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 480]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[PAD_CAST:%.+]] = builtin.unrealized_conversion_cast [[ZERO_PAD_VALUE]] : i8 to !qElemType
    // CHECK: [[PADDED_TENSOR:%.+]] = tensor.pad [[EXTRACTED_SLICE]] low[0, 0, [[PAD_LOW_FINAL_H]], [[PAD_LOW_FINAL_W]]] high[0, 0, 0, 0] {
    // CHECK-NEXT: ^bb0([[PAD_ARG_H:%.+]]: index, [[PAD_ARG_W:%.+]]: index, [[PAD_IXD_0:%.+]]: index, [[PAD_IXD_1:%.+]]: index):
    // CHECK-NEXT: tensor.yield [[PAD_CAST]] : !qElemType
    // CHECK: } : tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 90, 480]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x?x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 91, 481]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONVOLUTION_0:%.+]] = VPU.NCE.Convolution([[PADDED_TENSOR]], [[CONV1_WEIGHTS]])
    // CHECK-SAME: -> tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> 
    // CHECK: [[PAD_CAST_2:%.+]] = builtin.unrealized_conversion_cast [[PAD_VALUE:%.+]] : i8 to !qElemType5
    // CHECK: [[PADDED_TILE:%.+]] = tensor.pad [[CONVOLUTION_0]] low[0, 0, [[PAD_LOW_TEMP_H]], [[PAD_LOW_TEMP_W]]] high[0, 0, [[PAD_HIGH_TEMP_H]], [[PAD_HIGH_TEMP_W]]] {
    // CHECK-NEXT: ^bb0([[PAD_ARG_H]]: index, [[PAD_ARG_W]]: index, [[PAD_IXD_0]]: index, [[PAD_IXD_1]]: index):
    // CHECK-NEXT: tensor.yield [[PAD_CAST_2]] : !qElemType5
    // CHECK: } : tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?x!qElemType5, {bounds = #const.OpaqueI64Elements<[1, 32, 47, 242]> : tensor<4xsi64>, order = #NHWC}>    
    // CHECK: [[CONVOLUTION_1:%.+]] = VPU.NCE.Convolution([[PADDED_TILE]], [[CONV2_WEIGHTS]]) 
    // CHECK-SAME: -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> 
    // CHECK: [[INSERTED_SLICE:%.+]] = tensor.insert_slice [[CONVOLUTION_1]] into [[LOOP_ITER_3]][0, 0, [[LOOP_ITER]], [[LOOP_ITER_2]]] [1, 32, [[MIN_HEIGHT]], [[MIN_WIDTH]]] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 45, 240]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: scf.yield [[INSERTED_SLICE]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: scf.yield [[LOOP_W]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: return [[LOOP_H]] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

//CHECK: #[[$MAP0:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 48)>
//CHECK: #[[$MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0, 320)>
//CHECK: #[[$MAP2:.*]] = affine_map<(d0) -> (d0 floordiv 2)>


module @test {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
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

   %1 = VPU.NCE.Eltwise(%0, %0) {
        tilingStrategy = [1, 1, 1, 72],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, 
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} 
        -> tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> 
    
   return %1 : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> 

   // CHECK:   [[DIM_INDEX_W:%.+]] = arith.constant 3 : index
   // CHECK:   [[LOOP_STEP_W:%.+]] = arith.constant 320 : index
   // CHECK:   [[LOOP_STEP_H:%.+]] = arith.constant 48 : index
   // CHECK:   [[LOOP_BEGIN:%.+]] = arith.constant 0 : index

   // CHECK:   [[CONST_2:%.+]] = arith.constant 2 : index
   // CHECK:   [[DIM_H:%.+]] = tensor.dim [[INPUT]], [[CONST_2]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[DIM_H_MUL2:%.+]] = arith.muli [[DIM_H]], [[CONST_2]] : index
   // CHECK:   [[DIM_W:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[DIM_W_MUL2:%.+]] = arith.muli [[DIM_W]], [[CONST_2]] : index
    
   // CHECK:   [[LOOP_OUTPUT:%.+]] = tensor.empty([[DIM_H_MUL2]], [[DIM_W_MUL2]]) : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
   // CHECK:   [[DIM_H_1:%.+]] = tensor.dim [[INPUT]], [[CONST_2]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[LOOP_END_H:%.+]] = arith.muli [[DIM_H_1]], [[CONST_2]] : index
   // CHECK:   [[DIM_W_1:%.+]] = tensor.dim [[INPUT]], [[DIM_INDEX_W]] : tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
   // CHECK:   [[LOOP_END_W:%.+]] = arith.muli [[DIM_W_1]], [[CONST_2]] : index

   // CHECK:   [[LOOP_H:%.+]] = scf.for 
   // CHECK-SAME:              [[LOOP_ITER_H:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_H]] step [[LOOP_STEP_H]]
   // CHECK-SAME:              iter_args([[LOOP_OUT_H:%arg[0-9]]] = [[LOOP_OUTPUT]]) -> (tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>)

   // CHECK:                   [[LOOP_W:%.+]] = scf.for 
   // CHECK-SAME:                               [[LOOP_ITER_W:%arg[0-9]]] = [[LOOP_BEGIN]] to [[LOOP_END_W]] step [[LOOP_STEP_W]]
   // CHECK-SAME:                               iter_args([[LOOP_OUT_W:%arg[0-9]]] = [[LOOP_OUT_H]])
        
   // CHECK:                                    [[TMP_VALUE_H:%.+]] = affine.min #[[$MAP0]]([[LOOP_ITER_H]])[[[LOOP_END_H]]]
   // CHECK:                                    [[TMP_VALUE_W:%.+]] = affine.min #[[$MAP1]]([[LOOP_ITER_W]])[[[LOOP_END_W]]]
   // CHECK:                                    [[SLICE_OFFSET_H:%.+]] = affine.apply #[[$MAP2]]([[LOOP_ITER_H]])
   // CHECK:                                    [[SLICE_SIZE_H:%.+]] = affine.apply #[[$MAP2]]([[TMP_VALUE_H]])
   // CHECK:                                    [[SLICE_OFFSET_W:%.+]] = affine.apply #[[$MAP2]]([[LOOP_ITER_W]])
   // CHECK:                                    [[SLICE_SIZE_W:%.+]] = affine.apply #[[$MAP2]]([[TMP_VALUE_W]])

    // CHECK:                                   [[SLICE:%.+]] = tensor.extract_slice [[INPUT]][0, 0, [[SLICE_OFFSET_H]], [[SLICE_OFFSET_W]]] [1, 64, [[SLICE_SIZE_H]], [[SLICE_SIZE_W]]] [1, 1, 1, 1] 
    // CHECK-SAME:                              tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x64x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 64, 24, 160]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:                                   [[D2S:%.+]] = VPU.DepthToSpace([[SLICE]])
    // CHECK:                                   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:                                   [[INSERT:%.+]] = tensor.insert_slice [[ELTWISE]] into [[LOOP_OUT_W]][0, 0, [[LOOP_ITER_H]], [[LOOP_ITER_W]]] [1, 16, %7, %8] [1, 1, 1, 1] 
    // CHECK-SAME:                              tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 48, 320]> : tensor<4xsi64>, order = #NCHW}> into tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:   scf.yield [[INSERT]]
    
    // CHECK:   scf.yield [[LOOP_W]]
    // CHECK:   return [[LOOP_H]] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
}
}
