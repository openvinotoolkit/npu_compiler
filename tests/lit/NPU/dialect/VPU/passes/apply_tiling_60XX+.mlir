//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @NestedTilingUnrollChannelFirst
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @NestedTilingUnrollChannelFirst(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<256x1x1x1xf32> = dense<1.0> : tensor<256x1x1x1xf32>
    %bias_table = const.Declare tensor<256x1x1x1xf32> = dense<2.0> : tensor<256x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [256, 512, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1],
        tilingStrategy = [1, 4, 4, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x5x5xf16, {order = #NHWC}>, tensor<256x1x1x1xf32>, tensor<256x1x1x1xf32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // When the filter size * TOH * TOW size is bigger than the activation input size * TOC
    // unroll channel first, then H and W, to save the filter DMA
    // CHECK-DAG:       [[SCALETABLE_0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[BIASTABLE_0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[192, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALETABLE_1:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[BIASTABLE_1:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[128, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALETABLE_2:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[BIASTABLE_2:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[WEIGHTS_2:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[64, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[SCALETABLE_3:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[BIASTABLE_3:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:       [[WEIGHTS_3:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[0, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK:       [[SLICE_0_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_0:%.+]] = VPU.NCE.Convolution([[SLICE_0_0]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 0>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_1:%.+]] = VPU.NCE.Convolution([[SLICE_0_1]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 1>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_2:%.+]] = VPU.NCE.Convolution([[SLICE_0_2]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 2>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_3:%.+]] = VPU.NCE.Convolution([[SLICE_0_3]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 3>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_0:%.+]] = VPU.NCE.Convolution([[SLICE_1_0]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 0>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[SLICE_1_1]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 1>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_2:%.+]] = VPU.NCE.Convolution([[SLICE_1_2]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 2>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_3:%.+]] = VPU.NCE.Convolution([[SLICE_1_3]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 3>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_0:%.+]] = VPU.NCE.Convolution([[SLICE_2_0]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 0>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[SLICE_2_1]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 1>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_2:%.+]] = VPU.NCE.Convolution([[SLICE_2_2]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 2>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_3:%.+]] = VPU.NCE.Convolution([[SLICE_2_3]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 3>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_0:%.+]] = VPU.NCE.Convolution([[SLICE_3_0]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 0>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_1:%.+]] = VPU.NCE.Convolution([[SLICE_3_1]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 1>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_2:%.+]] = VPU.NCE.Convolution([[SLICE_3_2]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 2>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_3:%.+]] = VPU.NCE.Convolution([[SLICE_3_3]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 5, 5] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 3>}, pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:  strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV_0_0]], [[CONV_0_1]], [[CONV_0_2]], [[CONV_0_3]], [[CONV_1_0]], [[CONV_1_1]], [[CONV_1_2]], [[CONV_1_3]],
    // CHECK-SAME:        [[CONV_2_0]], [[CONV_2_1]], [[CONV_2_2]], [[CONV_2_3]], [[CONV_3_0]], [[CONV_3_1]], [[CONV_3_2]], [[CONV_3_3]])
    // CHECK-SAME:        [0, 0, 0, 0], [0, 0, 16, 0], [0, 0, 32, 0], [0, 0, 48, 0], [0, 64, 0, 0], [0, 64, 16, 0], [0, 64, 32, 0], [0, 64, 48, 0], [0, 128, 0, 0], [0, 128, 16, 0], [0, 128, 32, 0], [0, 128, 48, 0], [0, 192, 0, 0], [0, 192, 16, 0], [0, 192, 32, 0], [0, 192, 48, 0]
    // CHECK:       return [[CONCAT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @NestedTilingUnrollSpatialFirst
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x32x64xf16, {order = #NHWC}>
func.func @NestedTilingUnrollSpatialFirst(%arg0: tensor<1x512x32x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<256x1x1x1xf32> = dense<1.0> : tensor<256x1x1x1xf32>
    %bias_table = const.Declare tensor<256x1x1x1xf32> = dense<2.0> : tensor<256x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [256, 512, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,

        strides = [1, 1],
        tilingStrategy = [1, 4, 2, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x512x32x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x1xf32>, tensor<256x1x1x1xf32> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

    // When the filter size * TOH * TOW size is smaller than the activation input size * TOC
    // unroll H and W first, then C, to save the activation input DMA
    // CHECK-DAG:      [[SCALETABLE_0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[BIASTABLE_0:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[WEIGHTS_0:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[192, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[SCALETABLE_1:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[BIASTABLE_1:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[WEIGHTS_1:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[128, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[SCALETABLE_2:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[BIASTABLE_2:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[WEIGHTS_2:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[64, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[SCALETABLE_3:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[BIASTABLE_3:%.+]] = const.Declare tensor<64x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 1]>]
    // CHECK-DAG:      [[WEIGHTS_3:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK:          [[SLICE_0_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_0_0:%.+]] = VPU.NCE.Convolution([[SLICE_0_0]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 0>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_1_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_1_0:%.+]] = VPU.NCE.Convolution([[SLICE_1_0]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 0>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_2_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_2_0:%.+]] = VPU.NCE.Convolution([[SLICE_2_0]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 0>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_3_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_3_0:%.+]] = VPU.NCE.Convolution([[SLICE_3_0]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 0>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_0_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_0_1:%.+]] = VPU.NCE.Convolution([[SLICE_0_1]], [[WEIGHTS_3]], [[SCALETABLE_3]], [[BIASTABLE_3]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 0, 1>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_1_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[SLICE_1_1]], [[WEIGHTS_2]], [[SCALETABLE_2]], [[BIASTABLE_2]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 1, 1>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_2_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[SLICE_2_1]], [[WEIGHTS_1]], [[SCALETABLE_1]], [[BIASTABLE_1]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 2, 1>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_3_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_3_1:%.+]] = VPU.NCE.Convolution([[SLICE_3_1]], [[WEIGHTS_0]], [[SCALETABLE_0]], [[BIASTABLE_0]]) rawFilterShape [64, 512, 3, 3] {loop_tile_coord = {axes = array<i64: 1, 2>, indices = array<i64: 3, 1>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[CONCAT:%.+]] = VPU.Concat([[CONV_0_0]], [[CONV_1_0]], [[CONV_2_0]], [[CONV_3_0]], [[CONV_0_1]], [[CONV_1_1]], [[CONV_2_1]], [[CONV_3_1]])
    // CHECK-SAME:        [0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 192, 0, 0], [0, 0, 16, 0], [0, 64, 16, 0], [0, 128, 16, 0], [0, 192, 16, 0]
    // CHECK:          return [[CONCAT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: func.func @ApplyTilingNCEMatMulTileOverGroup
// CHECK-SAME:          [[INPUT0:%arg[0-9]]]: tensor<64x8x64x32xf16>, [[INPUT1:%arg[0-9]]]: tensor<64x8x64x32xf16>
func.func @ApplyTilingNCEMatMulTileOverGroup(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %scale_table = const.Declare tensor<1x64x1x1x1xf32> = dense<1.0> : tensor<1x64x1x1x1xf32>
  %bias_table = const.Declare tensor<1x64x1x1x1xf32> = dense<2.0> : tensor<1x64x1x1x1xf32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %scale_table, %bias_table) rawFilterShape [512, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK-DAG:           [[SCALE_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<1.000000e+00> : tensor<1x64x1x1x1xf32>
    // CHECK-DAG:           [[BIAS_TABLE:%.+]] = const.Declare tensor<1x64x1x1x1xf32> = dense<2.000000e+00> : tensor<1x64x1x1x1xf32>
    // CHECK:               [[INPUT0_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs([[INPUT0]] : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
    // CHECK:               [[INPUT1_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs([[INPUT1]] : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
    // CHECK:               [[INPUT0_AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[INPUT0_SHAPE_CAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]}
    // CHECK-SAME:              tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
    // CHECK:               [[INPUT0_PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT0_AFFINE_RESHAPE]]) {dst_order = #GNHWC, mem_perm = #map} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
    // CHECK:               [[INPUT1_AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[INPUT1_SHAPE_CAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]}
    // CHECK-SAME:              tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
    // CHECK:               [[INPUT1_PERMUTE_CAST:%.+]] = VPU.PermuteCast([[INPUT1_AFFINE_RESHAPE]]) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>

    // this reshape here is an optimization for better compute stencil match
    // CHECK:               [[INPUT0_AFFINE_RESHAPE2:%.+]] = VPU.AffineReshape([[INPUT0_PERMUTE_CAST]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]}
    // CHECK-SAME:              tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>

    // Slice 0
    // CHECK:               [[SLICE0_INPUT0:%.+]] = VPU.Slice [[INPUT0_AFFINE_RESHAPE2]] [0, 0, 0, 0, 0] [256, 1, 32, 16, 4] : tensor<512x1x32x16x4xf16, {order = #GNHWC}> to tensor<256x1x32x16x4xf16, {order = #GNHWC}>
    // CHECK:               [[SLICE0_INPUT1:%.+]] = VPU.Slice [[INPUT1_PERMUTE_CAST]] [0, 0, 0, 0, 0] [256, 64, 32, 1, 1] : tensor<512x64x32x1x1xf16, {order = #GNHWC}> to tensor<256x64x32x1x1xf16, {order = #GNHWC}>

    // CHECK:               [[MATMUL_SLICE0:%.+]] = VPU.NCE.MatMul([[SLICE0_INPUT0]], [[SLICE0_INPUT1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [256, 64, 32, 1, 1] {
    // CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:               strides = [1, 1]
    // CHECK-SAME:               tensor<256x1x64x16x4xf16, {order = #GNHWC}>

    // Slice 1
    // CHECK:               [[SLICE1_INPUT0:%.+]] = VPU.Slice [[INPUT0_AFFINE_RESHAPE2]] [256, 0, 0, 0, 0] [256, 1, 32, 16, 4] : tensor<512x1x32x16x4xf16, {order = #GNHWC}> to tensor<256x1x32x16x4xf16, {order = #GNHWC}>
    // CHECK:               [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_PERMUTE_CAST]] [256, 0, 0, 0, 0] [256, 64, 32, 1, 1] : tensor<512x64x32x1x1xf16, {order = #GNHWC}> to tensor<256x64x32x1x1xf16, {order = #GNHWC}>

    // CHECK:               [[MATMUL_SLICE1:%.+]] = VPU.NCE.MatMul([[SLICE1_INPUT0]], [[SLICE1_INPUT1]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [256, 64, 32, 1, 1] {
    // CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:               strides = [1, 1]
    // CHECK-SAME:               tensor<256x1x64x16x4xf16, {order = #GNHWC}>

    // Concat
    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[MATMUL_SLICE0]], [[MATMUL_SLICE1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0, 0], [256, 0, 0, 0, 0]]}
    // CHECK-SAME:          tensor<256x1x64x16x4xf16, {order = #GNHWC}>, tensor<256x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
    // CHECK:               [[OUTPUT_RESHAPE:%.+]] = VPU.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]}
    // CHECK-SAME:              tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
    // CHECK:               return [[OUTPUT_RESHAPE]] : tensor<512x1x64x64x1xf16, {order = #GNHWC}>
}

// -----

// CHECK-LABEL: @TileSoftMaxWithMax
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x4x256x256xf16>, [[MAX:%.+]]: tensor<1x1x256x256xf16>)
func.func @TileSoftMaxWithMax(%input: tensor<1x4x256x256xf16>, %max: tensor<1x1x256x256xf16>) -> tensor<1x4x256x256xf16> {
    %0 = VPU.SoftMax(%input, %max) {axisInd = 1 : i64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x4x256x256xf16>, tensor<1x1x256x256xf16> -> tensor<1x4x256x256xf16>
    return %0 : tensor<1x4x256x256xf16>

    // CHECK: [[SLICE_IN_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 128, 256]
    // CHECK-SAME: tensor<1x4x256x256xf16> to tensor<1x4x128x256xf16>
    // CHECK: [[SLICE_MAX_0:%.+]] = VPU.Slice [[MAX]] [0, 0, 0, 0] [1, 1, 128, 256]
    // CHECK-SAME: tensor<1x1x256x256xf16> to tensor<1x1x128x256xf16>
    // CHECK: [[SOFTMAX_0:%.+]] = VPU.SoftMax([[SLICE_IN_0]], [[SLICE_MAX_0]]) {axisInd = 1 : i64
    // CHECK-SAME: tensor<1x4x128x256xf16>, tensor<1x1x128x256xf16> -> tensor<1x4x128x256xf16>
    // CHECK: [[SLICE_IN_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 4, 128, 256]
    // CHECK-SAME: tensor<1x4x256x256xf16> to tensor<1x4x128x256xf16>
    // CHECK: [[SLICE_MAX_1:%.+]] = VPU.Slice [[MAX]] [0, 0, 128, 0] [1, 1, 128, 256]
    // CHECK-SAME: tensor<1x1x256x256xf16> to tensor<1x1x128x256xf16>
    // CHECK: [[SOFTMAX_1:%.+]] = VPU.SoftMax([[SLICE_IN_1]], [[SLICE_MAX_1]]) {axisInd = 1 : i64
    // CHECK-SAME: tensor<1x4x128x256xf16>, tensor<1x1x128x256xf16> -> tensor<1x4x128x256xf16>
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[SOFTMAX_0]], [[SOFTMAX_1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 0, 128, 0]]}
    // CHECK-SAME: tensor<1x4x128x256xf16>, tensor<1x4x128x256xf16> -> tensor<1x4x256x256xf16>
    // CHECK: return [[CONCAT]] : tensor<1x4x256x256xf16>
}
