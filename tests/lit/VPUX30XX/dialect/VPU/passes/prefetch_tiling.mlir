//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --prefetch-tiling --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @NoTilingConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<128x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x128x1x1xf16>
func @NoTilingConv(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<128x32x3x3xf16>,
        %bias: tensor<1x128x1x1xf16>)
            -> tensor<1x128x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<128x32x3x3xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x64x64xf16>
    return %1 : tensor<1x128x64x64xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16>
}

// -----

// CHECK-LABEL: func @NoTilingMaxPool
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x170x170xf16>
func @NoTilingMaxPool(
        %input: tensor<1x16x170x170xf16>)
            -> tensor<1x16x170x170xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x170x170xf16> -> tensor<1x16x170x170xf16>
    return %1 : tensor<1x16x170x170xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x170x170xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x170x170xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<256x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<256x1x1x4xsi32, {order = #NHWC}>
func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<256x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<256x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = "ADD"} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<256x144x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NHWC}> -> tensor<1x256x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS2]] [0, 0, 0, 0] [128, 144, 3, 3]
    // CHECK-SAME:      tensor<256x144x3x3xf16, {order = #NHWC}> to tensor<128x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE0:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [0, 0, 0, 0] [128, 1, 1, 4]
    // CHECK-SAME:      tensor<256x1x1x4xsi32, {order = #NHWC}> to tensor<128x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x128x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS2]] [128, 0, 0, 0] [128, 144, 3, 3]
    // CHECK-SAME:      tensor<256x144x3x3xf16, {order = #NHWC}> to tensor<128x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE1:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [128, 0, 0, 0] [128, 1, 1, 4]
    // CHECK-SAME:      tensor<256x1x1x4xsi32, {order = #NHWC}> to tensor<128x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x128x20x20xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x170x170xf16, {order = #NHWC}>)
func @SplitNCEPoolOverH(%arg0: tensor<1x16x170x170xf16, {order = #NHWC}>) -> tensor<1x16x170x170xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x170x170xf16, {order = #NHWC}>

    return %0 : tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 44, 170]
    // CHECK-SAME:      : tensor<1x16x170x170xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x44x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:      } -> tensor<1x16x43x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 16, 45, 170]
    // CHECK-SAME:      : tensor<1x16x170x170xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x45x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x43x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 85, 0] [1, 16, 44, 170]
    // CHECK-SAME:      : tensor<1x16x170x170xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x44x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x42x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 127, 0] [1, 16, 43, 170]
    // CHECK-SAME:      : tensor<1x16x170x170xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x43x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x42x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 128, 0]
    // CHECK-SAME:      -> tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x170x170xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func @NoTileWithSOH(
        %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
            -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1>
        : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>
func @TileWithSOH(
        %arg0: tensor<1x16x200x200xf16, {order = #NHWC}>)
            -> tensor<1x32x200x200xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
        : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x32x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x32x200x200xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:          tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x100x200xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:          tensor<1x16x200x200xf16, {order = #NHWC}> to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x100x200xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x32x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
func @NoTileWithSOK(
        %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
            -> tensor<1x240x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<240x1x1x4xsi32> = dense<1>
        : tensor<240x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func @TileWithSOK(
        %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
            -> tensor<1x384x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<384x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<384x1x1x4xsi32> = dense<1>
        : tensor<384x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [384, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x384x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<192x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[192, 0, 0, 0], [192, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<192x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[192, 0, 0, 0], [192, 32, 7, 7]>
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<192x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [192, 1, 1, 4]>
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<192x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [192, 32, 7, 7]>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [192, 32, 7, 7]
    // CHECK-SAME:          tensor<1x192x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [192, 32, 7, 7]
    // CHECK-SAME:          tensor<1x192x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x384x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func @LargeConstPipeliningSOK(
        %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
            -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1>
        : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>


    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>]
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [256, 256, 3, 3]>]

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x20x20xf16, {order = #NHWC}>
func @SplitNCEEltwise(
        %arg0: tensor<1x512x20x20xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x20x20xf16, {order = #NHWC}>)
            -> tensor<1x512x20x20xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x512x20x20xf16, {order = #NHWC}>

    return %0 : tensor<1x512x20x20xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[INPUT_0_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 256, 20, 20]
    // CHECK-SAME:      : tensor<1x512x20x20xf16, {order = #NHWC}> to tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 256, 20, 20]
    // CHECK-SAME:      : tensor<1x512x20x20xf16, {order = #NHWC}> to tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0_0]], [[INPUT_1_0]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[INPUT_0_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 256, 0, 0] [1, 256, 20, 20]
    // CHECK-SAME:      : tensor<1x512x20x20xf16, {order = #NHWC}> to tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 256, 0, 0] [1, 256, 20, 20]
    // CHECK-SAME:      : tensor<1x512x20x20xf16, {order = #NHWC}> to tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[INPUT_0_1]], [[INPUT_1_1]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>

    // Concat
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      : tensor<1x256x20x20xf16, {order = #NHWC}>, tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x20x20xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x512x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x48x48xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x48x48xf16, {order = #NHWC}>
func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x48x48xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x48x48xf16, {order = #NHWC}>)
            -> tensor<1x64x48x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x48x48xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x64x48x48xf16, {order = #NHWC}>

    return %1 : tensor<1x64x48x48xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x48x48xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK-NOT:   VPU.Slice
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x64x48x48xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x48x48xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x128x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00> : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<128x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<128x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<128x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<128x1x1x384xi1>, is_weights

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x128x32x64xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x128x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x3840xf16, {order = #NHWC}>
func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x3840xf16, {order = #NHWC}>) -> tensor<1x16x1x3840xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}, strides = [1, 1]} -> tensor<1x16x1x3840xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x3840xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 1280]
    // CHECK-SAME:      : tensor<1x16x7x3840xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x1280xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x1280xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1280] [1, 16, 7, 1280]
    // CHECK-SAME:      : tensor<1x16x7x3840xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x1280xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x1280xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2560] [1, 16, 7, 1280]
    // CHECK-SAME:      : tensor<1x16x7x3840xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x1280xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x1280xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 1280], [0, 0, 0, 2560]
    // CHECK-SAME:      -> tensor<1x16x1x3840xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x3840xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitAveragePoolNoTileOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x7680xf16, {order = #NHWC}>
func @SplitAveragePoolNoTileOverW(%arg0: tensor<1x16x7x7680xf16, {order = #NHWC}>) -> tensor<1x16x1x7680xf16, {order = #NHWC}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x16x7x7680xf16, {order = #NHWC}> -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x7680xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [7, 1],
    // CHECK-SAME:      pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x7x7680xf16, {order = #NHWC}> -> tensor<1x16x1x7680xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x7680xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x18x3000xf16, {order = #NHWC}>
func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x18x3000xf16, {order = #NHWC}>) -> tensor<1x16x18x3000xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1],
        multiClusterStrategy = "SplitOverHeight"
    } -> tensor<1x16x18x3000xf16, {order = #NHWC}>

    return %0 : tensor<1x16x18x3000xf16, {order = #NHWC}>

    // CHECK:       [[ACT_WIN:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile 0
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 18, 1501]
    // CHECK-SAME:      tensor<1x16x18x3000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x18x1501xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[SLICE_0]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x16x18x1500xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1499] [1, 16, 18, 1501]
    // CHECK-SAME:      tensor<1x16x18x3000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x18x1501xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[SLICE_1]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x16x18x1500xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
    // CHECK:       return [[CONCAT]] : tensor<1x16x18x3000xf16, {order = #NHWC}>
}
