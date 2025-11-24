//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @NestedTilingUnrollChannelFirst
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @NestedTilingUnrollChannelFirst(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 512, 5, 5],
        strides = [1, 1],
        tilingStrategy = [1, 4, 4, 1]
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x5x5xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // When the filter size * TOH * TOW size is bigger than the activation input size * TOC
    // unroll channel first, then H and W, to save the filter DMA
    // CHECK-DAG:       [[WT_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[192, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WT_1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[128, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WT_2:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS_2:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[64, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WT_3:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS_3:%.+]] = const.Declare tensor<64x512x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x5x5xf16>, [#const.SubView<[0, 0, 0, 0], [64, 512, 5, 5]>, #const.Reorder<#NHWC>]
    // CHECK:       [[SLICE_0_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_0:%.+]] = VPU.NCE.Convolution([[SLICE_0_0]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_1:%.+]] = VPU.NCE.Convolution([[SLICE_0_1]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_2:%.+]] = VPU.NCE.Convolution([[SLICE_0_2]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0_3:%.+]] = VPU.NCE.Convolution([[SLICE_0_3]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_0:%.+]] = VPU.NCE.Convolution([[SLICE_1_0]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[SLICE_1_1]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_2:%.+]] = VPU.NCE.Convolution([[SLICE_1_2]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_1_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1_3:%.+]] = VPU.NCE.Convolution([[SLICE_1_3]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_0:%.+]] = VPU.NCE.Convolution([[SLICE_2_0]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[SLICE_2_1]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_2:%.+]] = VPU.NCE.Convolution([[SLICE_2_2]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_2_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2_3:%.+]] = VPU.NCE.Convolution([[SLICE_2_3]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_0:%.+]] = VPU.NCE.Convolution([[SLICE_3_0]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_1:%.+]] = VPU.NCE.Convolution([[SLICE_3_1]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 30, 0] [1, 512, 20, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x20x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_2:%.+]] = VPU.NCE.Convolution([[SLICE_3_2]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
    // CHECK-SAME:       -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_3_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 46, 0] [1, 512, 18, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x18x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV_3_3:%.+]] = VPU.NCE.Convolution([[SLICE_3_3]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 5, 5], strides = [1, 1]}
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
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 512, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 4, 2, 1]
    } : tensor<1x512x32x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

    // When the filter size * TOH * TOW size is smaller than the activation input size * TOC
    // unroll H and W first, then C, to save the activation input DMA
    // CHECK-DAG:      [[WT_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:      [[WEIGHTS_0:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[192, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[WT_1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:      [[WEIGHTS_1:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[128, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[WT_2:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:      [[WEIGHTS_2:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[64, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:      [[WT_3:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]
    // CHECK-DAG:      [[WEIGHTS_3:%.+]] = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [64, 512, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK:          [[SLICE_0_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_0_0:%.+]] = VPU.NCE.Convolution([[SLICE_0_0]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_1_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_1_0:%.+]] = VPU.NCE.Convolution([[SLICE_1_0]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_2_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_2_0:%.+]] = VPU.NCE.Convolution([[SLICE_2_0]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_3_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_3_0:%.+]] = VPU.NCE.Convolution([[SLICE_3_0]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_0_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_0_1:%.+]] = VPU.NCE.Convolution([[SLICE_0_1]], [[WEIGHTS_3]], [[WT_3]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_1_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_1_1:%.+]] = VPU.NCE.Convolution([[SLICE_1_1]], [[WEIGHTS_2]], [[WT_2]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_2_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_2_1:%.+]] = VPU.NCE.Convolution([[SLICE_2_1]], [[WEIGHTS_1]], [[WT_1]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x64x16x64xf16, {order = #NHWC}>
    // CHECK:          [[SLICE_3_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
    // CHECK:          [[CONV_3_1:%.+]] = VPU.NCE.Convolution([[SLICE_3_1]], [[WEIGHTS_0]], [[WT_0]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]}
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
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 64, 32, 1, 1], strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:               [[WT_1:%.+]] = const.Declare tensor<256x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>, [#const.SubView<[256, 0, 0, 0, 0], [256, 64, 1, 1, 4]>]
    // CHECK:               [[WT_0:%.+]] = const.Declare tensor<256x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0, 0], [256, 64, 1, 1, 4]>]
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

    // CHECK:               [[MATMUL_SLICE0:%.+]] = VPU.NCE.MatMul([[SLICE0_INPUT0]], [[SLICE0_INPUT1]], [[WT_0]])
    // CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:               rawFilterShape = [256, 64, 32, 1, 1]
    // CHECK-SAME:               strides = [1, 1]
    // CHECK-SAME:               tensor<256x1x64x16x4xf16, {order = #GNHWC}>

    // Slice 1
    // CHECK:               [[SLICE1_INPUT0:%.+]] = VPU.Slice [[INPUT0_AFFINE_RESHAPE2]] [256, 0, 0, 0, 0] [256, 1, 32, 16, 4] : tensor<512x1x32x16x4xf16, {order = #GNHWC}> to tensor<256x1x32x16x4xf16, {order = #GNHWC}>
    // CHECK:               [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_PERMUTE_CAST]] [256, 0, 0, 0, 0] [256, 64, 32, 1, 1] : tensor<512x64x32x1x1xf16, {order = #GNHWC}> to tensor<256x64x32x1x1xf16, {order = #GNHWC}>

    // CHECK:               [[MATMUL_SLICE1:%.+]] = VPU.NCE.MatMul([[SLICE1_INPUT0]], [[SLICE1_INPUT1]], [[WT_1]])
    // CHECK-SAME:               multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:               rawFilterShape = [256, 64, 32, 1, 1]
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
