//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --cmx-stack-frames-reserve-mem --cmx-metadata-reserve-mem --tiling="prefetching=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOHSplitWeightTableFormat
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x96x48xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOHSplitWeightTableFormat(%arg0: tensor<1x32x96x48xf16, {order = #NHWC}>) -> tensor<1x256x96x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<256x1x1x1xf32> = dense<1.0> : tensor<256x1x1x1xf32>
    %bias_table = const.Declare tensor<256x1x1x1xf32> = dense<2.0> : tensor<256x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x32x96x48xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x1xf32>, tensor<256x1x1x1xf32> -> tensor<1x256x96x48xf16, {order = #NHWC}>

    return %0 : tensor<1x256x96x48xf16, {order = #NHWC}>

    // CHECK-DAG:        [[SCALE_TABLE:%.+]] = const.Declare tensor<256x1x1x1xf32> = dense<1.000000e+00> : tensor<256x1x1x1xf32>
    // CHECK-DAG:        [[BIAS_TABLE:%.+]] = const.Declare tensor<256x1x1x1xf32> = dense<2.000000e+00> : tensor<256x1x1x1xf32>
    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 49, 48]
    // CHECK-SAME:      : tensor<1x32x96x48xf16, {order = #NHWC}> to tensor<1x32x49x48xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [256, 32, 3, 3] {
    // CHECK-SAME:          loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64
    // CHECK-SAME:          -> tensor<1x256x48x48xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 47, 0] [1, 32, 49, 48]
    // CHECK-SAME:      : tensor<1x32x96x48xf16, {order = #NHWC}> to tensor<1x32x49x48xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [256, 32, 3, 3] {
    // CHECK-SAME:          loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          strides = [1, 1], tiling_loop_index = 0 : i64}
    // CHECK-SAME:          -> tensor<1x256x48x48xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 48, 0]
    // CHECK-SAME:          -> tensor<1x256x96x48xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x96x48xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

// CHECK-LABEL:   @SplitI4QuantNCEConvOverOCSplitWeightTableFormat
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
func.func @SplitI4QuantNCEConvOverOCSplitWeightTableFormat(%arg0: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<6320x1x1x1xf32, {order = #NCHW}> = dense<1.0> : tensor<6320x1x1x1xf32>
    %bias_table = const.Declare tensor<6320x1x1x1xf32, {order = #NCHW}> = dense<2.0> : tensor<6320x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [6320, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
         strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x1xf32, {order = #NCHW}>, tensor<6320x1x1x1xf32, {order = #NCHW}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:           [[WEIGHTS_0:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK:           [[SCALE_TABLE_0:%.+]] = const.Declare tensor<6320x1x1x1xf32, {order = #NCHW}> = dense<1.000000e+00> : tensor<6320x1x1x1xf32>
    // CHECK:           [[BIAS_TABLE_0:%.+]] = const.Declare tensor<6320x1x1x1xf32, {order = #NCHW}> = dense<2.000000e+00> : tensor<6320x1x1x1xf32>
    // CHECK:           [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 128, 86, 4] : tensor<1x128x256x4xf16, {order = #NHWC}> to tensor<1x128x86x4xf16, {order = #NHWC}>
    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE_0]], [[WEIGHTS_0]], [[SCALE_TABLE_0]], [[BIAS_TABLE_0]]) rawFilterShape [6320, 128, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 0>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1], tiling_loop_index = 0 : i64} : tensor<1x128x86x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x1xf32, {order = #NCHW}>, tensor<6320x1x1x1xf32, {order = #NCHW}> -> tensor<1x6320x86x4xf16, {order = #NHWC}>
    // CHECK:           [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 86, 0] [1, 128, 85, 4] : tensor<1x128x256x4xf16, {order = #NHWC}> to tensor<1x128x85x4xf16, {order = #NHWC}>
    // CHECK:           [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE_1]], [[WEIGHTS_0]], [[SCALE_TABLE_0]], [[BIAS_TABLE_0]]) rawFilterShape [6320, 128, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 1>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1], tiling_loop_index = 0 : i64} : tensor<1x128x85x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x1xf32, {order = #NCHW}>, tensor<6320x1x1x1xf32, {order = #NCHW}> -> tensor<1x6320x85x4xf16, {order = #NHWC}>
    // CHECK:           [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 171, 0] [1, 128, 85, 4] : tensor<1x128x256x4xf16, {order = #NHWC}> to tensor<1x128x85x4xf16, {order = #NHWC}>
    // CHECK:           [[CONV_2:%.+]] = VPU.NCE.Convolution([[SLICE_2]], [[WEIGHTS_0]], [[SCALE_TABLE_0]], [[BIAS_TABLE_0]]) rawFilterShape [6320, 128, 1, 1] {loop_tile_coord = {axes = array<i64: 2>, indices = array<i64: 2>}, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1], tiling_loop_index = 0 : i64} : tensor<1x128x85x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x1xf32, {order = #NCHW}>, tensor<6320x1x1x1xf32, {order = #NCHW}> -> tensor<1x6320x85x4xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
}
