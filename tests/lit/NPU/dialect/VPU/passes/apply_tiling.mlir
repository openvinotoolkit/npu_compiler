//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ApplyTilingNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @ApplyTilingNCEConv(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingMaxPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @ApplyTilingMaxPool(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NCHW}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]] ) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]] ) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingAvgPool
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x12960xf16, {order = #NHWC}>
func.func @ApplyTilingAvgPool(%arg0: tensor<1x16x7x12960xf16, {order = #NHWC}>) -> tensor<1x16x1x12960xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [7, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        tilingStrategy = [1, 1, 1, 3]
        } -> tensor<1x16x1x12960xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x12960xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]])
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4320] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]])
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 8640] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]])
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 4320], [0, 0, 0, 8640]
    // CHECK-SAME:      -> tensor<1x16x1x12960xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x12960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingNCEEltwise
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func.func @ApplyTilingNCEEltwise(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 2, 1, 1]
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ApplyTilingSparseNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
func.func @ApplyTilingSparseNCEConv(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights> -> tensor<1x160x80x60xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_TILE:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]


    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_SM_TILE]], [[WEIGHTS_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 60]
    // CHECK-SAME:      : tensor<1x32x80x60xf16, {order = #NHWC}> to tensor<1x32x41x60xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x60xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 60]
    // CHECK-SAME:      : tensor<1x32x80x60xf16, {order = #NHWC}> to tensor<1x32x41x60xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_2]], [[WEIGHTS_SPARSE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x60xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @ApplyTilingSparseQuantNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType, {order = #NHWC}>
func.func @ApplyTilingSparseQuantNCEConv(%arg0: tensor<1x32x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>
    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x32x80x80x!qElemType, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights> -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]


    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType, {order = #NHWC}> to tensor<1x32x41x80x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_0]], [[WEIGHTS_SPARSE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType, {order = #NHWC}> to tensor<1x32x41x80x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ApplyTilingDepthToSpace
func.func @ApplyTilingDepthToSpace(%arg0: tensor<1x384x10x120xf16, {order = #NHWC}>) -> tensor<1x24x40x480xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {
        block_size = 4 : i64,
        mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x384x10x120xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>
    return %0 : tensor<1x24x40x480xf16, {order = #NHWC}>

    // CHECK:           [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 384, 5, 120] : tensor<1x384x10x120xf16, {order = #NHWC}> to tensor<1x384x5x120xf16, {order = #NHWC}>
    // CHECK:           [[D2S0:%.+]] = VPU.DepthToSpace([[SLICE0]])
    // CHECK-SAME:              block_size = 4 : i64
    // CHECK-SAME:              mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    // CHECK-SAME:          : tensor<1x384x5x120xf16, {order = #NHWC}> -> tensor<1x24x20x480xf16, {order = #NHWC}>

    // CHECK:           [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 0, 5, 0] [1, 384, 5, 120] : tensor<1x384x10x120xf16, {order = #NHWC}> to tensor<1x384x5x120xf16, {order = #NHWC}>
    // CHECK:           [[D2S1:%.+]] = VPU.DepthToSpace([[SLICE1]])
    // CHECK-SAME:              block_size = 4 : i64
    // CHECK-SAME:              mode = #IE.depth_to_space_mode<BLOCKS_FIRST>
    // CHECK-SAME:          : tensor<1x384x5x120xf16, {order = #NHWC}> -> tensor<1x24x20x480xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT:%.+]] = VPU.Concat([[D2S0]], [[D2S1]])
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 20, 0]]
    // CHECK-SAME:          : tensor<1x24x20x480xf16, {order = #NHWC}>, tensor<1x24x20x480xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>

    // CHECK:           return [[CONCAT]] : tensor<1x24x40x480xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @ApplyTilingSpaceToDepth
func.func @ApplyTilingSpaceToDepth(%arg0: tensor<1x48x160x80xf16>) -> tensor<1x768x40x20xf16> {
    %0 = VPU.SpaceToDepthOp(%arg0) {
        block_size = 4 : i64,
        mode = #IE.space_to_depth_mode<BLOCKS_FIRST>,
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    return %0 : tensor<1x768x40x20xf16>

    // CHECK:           [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:           [[S2D0:%.+]] = VPU.SpaceToDepthOp([[SLICE0]])
    // CHECK-SAME:              block_size = 4 : i64
    // CHECK-SAME:              mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    // CHECK-SAME:          : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:           [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 40] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:           [[S2D1:%.+]] = VPU.SpaceToDepthOp([[SLICE1]])
    // CHECK-SAME:              block_size = 4 : i64
    // CHECK-SAME:              mode = #IE.space_to_depth_mode<BLOCKS_FIRST>
    // CHECK-SAME:          : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:           [[CONCAT:%.+]] = VPU.Concat([[S2D0]], [[S2D1]])
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 10]]
    // CHECK-SAME:          : tensor<1x768x40x10xf16>, tensor<1x768x40x10xf16> -> tensor<1x768x40x20xf16>

    // CHECK:           return [[CONCAT]] : tensor<1x768x40x20xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ApplyTilingConvert
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x3000xui8, {order = #NHWC}>
func.func @ApplyTilingConvert(%arg0: tensor<1x2x80x3000xui8, {order = #NHWC}>) -> tensor<1x2x80x3000xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f32,
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x2x80x3000xui8, {order = #NHWC}> -> tensor<1x2x80x3000xf32, {order = #NHWC}>
    return %0 : tensor<1x2x80x3000xf32, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 80, 1500]
    // CHECK-SAME:      : tensor<1x2x80x3000xui8, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x2x80x1500xui8, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convert([[INPUT_TILE0]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      }> -> tensor<1x2x80x1500xf32, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1500] [1, 2, 80, 1500]
    // CHECK-SAME:      : tensor<1x2x80x3000xui8, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x2x80x1500xui8, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convert([[INPUT_TILE1]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      }> -> tensor<1x2x80x1500xf32, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 1500]
    // CHECK-SAME:      -> tensor<1x2x80x3000xf32, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x2x80x3000xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ApplyTilingSigmoid
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingSigmoid(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Sigmoid(%arg0) {
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sigmoid([[INPUT_TILE0]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sigmoid([[INPUT_TILE1]])
    // CHECK-SAME:       : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingTanh
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingTanh(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Tanh(%arg0) {
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tanh([[INPUT_TILE0]])
    // CHECK-SAME:       : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:       : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tanh([[INPUT_TILE1]])
    // CHECK-SAME:       : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:       : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingExp
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingExp(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Exp(%arg0) {
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:       : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Exp([[INPUT_TILE0]])
    // CHECK-SAME:       : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:       : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Exp([[INPUT_TILE1]])
    // CHECK-SAME:       : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:       : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingSqrt
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingSqrt(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Sqrt(%arg0) {
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:      : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sqrt([[INPUT_TILE0]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:      : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sqrt([[INPUT_TILE1]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:      : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingClamp
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingClamp(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Clamp(%arg0) {
        max = 1.000000e+00 : f64,
        min = -1.000000e+00 : f64,
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:      : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingReLU
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ApplyTilingReLU(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.ReLU(%arg0) {
        tilingStrategy = [1, 1, 1, 2]
    } : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:      : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:      : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
    // CHECK-SAME:      : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:      : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingHSwish
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16>
func.func @ApplyTilingHSwish(%arg0: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16> {
    %0 = VPU.HSwish(%arg0) {
        tilingStrategy = [1, 1, 1, 4]
    } : tensor<1x16x80x960xf16> -> tensor<1x16x80x960xf16>
    return %0 : tensor<1x16x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 80, 240]
    // CHECK-SAME:      : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSwish([[INPUT_TILE0]])
    // CHECK-SAME:      : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 240] [1, 16, 80, 240]
    // CHECK-SAME:      : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSwish([[INPUT_TILE1]])
    // CHECK-SAME:      : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 16, 80, 240]
    // CHECK-SAME:  : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.HSwish([[INPUT_TILE2]])
    // CHECK-SAME:      : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 720] [1, 16, 80, 240]
    // CHECK-SAME:      : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.HSwish([[INPUT_TILE3]])
    // CHECK-SAME:      : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 240], [0, 0, 0, 480], [0, 0, 0, 720]
    // CHECK-SAME:      : tensor<1x16x80x240xf16>, tensor<1x16x80x240xf16>, tensor<1x16x80x240xf16>, tensor<1x16x80x240xf16> -> tensor<1x16x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x960xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingDivide
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
func.func @ApplyTilingDivide(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        tilingStrategy = [1, 1, 2, 1]
    } : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
    return %0 : tensor<1x10x256x176xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
    // CHECK-SAME:      : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
    // CHECK-SAME:      : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Divide([[INPUT_TILE0]], [[INPUT_TILE1]])
    // CHECK-SAME:      : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
    // CHECK-SAME:      : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
    // CHECK-SAME:      : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Divide([[INPUT_TILE2]], [[INPUT_TILE3]])
    // CHECK-SAME:      : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 128, 0]
    // CHECK-SAME:      : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
}

// -----

// CHECK:      [[INPUT_0:%arg[0-9]]]: tensor<1x64x112x112xf16>, [[INPUT_1:%arg[0-9]]]: tensor<64x2x3x3xf16>) -> tensor<1x64x56x56xf16>
func.func @SplitSwGroupConvOverOC(
        %input: tensor<1x64x112x112xf16>,
        %filter: tensor<64x2x3x3xf16>)
            -> tensor<1x64x56x56xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2],
        groups = 32 : i64,
        tilingStrategy = [1, 32, 1, 1]

    } : tensor<1x64x112x112xf16>, tensor<64x2x3x3xf16> -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 2, 112, 112]
    // CHECK-SAME:      : tensor<1x64x112x112xf16> to tensor<1x2x112x112xf16>
    // CHECK:        [[WEIGHTS_TILE_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [2, 2, 3, 3]
    // CHECK-SAME:      : tensor<64x2x3x3xf16> to tensor<2x2x3x3xf16>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_0]], [[WEIGHTS_TILE_0]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x2x56x56xf16>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 2, 0, 0] [1, 2, 112, 112]
    // CHECK-SAME:      : tensor<1x64x112x112xf16> to tensor<1x2x112x112xf16>
    // CHECK:        [[WEIGHTS_TILE_1:%.+]] = VPU.Slice [[INPUT_1]] [2, 0, 0, 0] [2, 2, 3, 3]
    // CHECK-SAME:      : tensor<64x2x3x3xf16> to tensor<2x2x3x3xf16>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_1]], [[WEIGHTS_TILE_1]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x2x56x56xf16>

    // CHECK:        [[ACTIVATION_TILE_31:%.+]] = VPU.Slice [[INPUT_0]] [0, 62, 0, 0] [1, 2, 112, 112]
    // CHECK-SAME:      : tensor<1x64x112x112xf16> to tensor<1x2x112x112xf16>
    // CHECK:        [[WEIGHTS_TILE_31:%.+]] = VPU.Slice [[INPUT_1]] [62, 0, 0, 0] [2, 2, 3, 3]
    // CHECK-SAME:      : tensor<64x2x3x3xf16> to tensor<2x2x3x3xf16>

    // CHECK:        [[OUTPUT_TILE31:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_31]], [[WEIGHTS_TILE_31]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x2x56x56xf16>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]]
    // CHECK-SAME:          [[OUTPUT_TILE31]]
    // CHECK-SAME:          [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]
    // CHECK-SAME:          [0, 58, 0, 0], [0, 60, 0, 0], [0, 62, 0, 0]
    // CHECK-SAME:          -> tensor<1x64x56x56xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x56x56xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwGroupConvOverOCLessOCThanGroup
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x10x224x224xf16>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<4x5x5x5xf16>
func.func @SplitSwGroupConvOverOCLessOCThanGroup(
        %input: tensor<1x10x224x224xf16>,
        %filter: tensor<4x5x5x5xf16>)
            -> tensor<1x4x224x224xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [2, 2],
        pads_end = [2, 2],
        strides = [1, 1],
        groups = 2 : i64,
        tilingStrategy = [1, 4, 1, 1]

    } : tensor<1x10x224x224xf16>, tensor<4x5x5x5xf16> -> tensor<1x4x224x224xf16>
    return %1 : tensor<1x4x224x224xf16>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 224, 224]
    // CHECK-SAME:      : tensor<1x10x224x224xf16> to tensor<1x5x224x224xf16>
    // CHECK:        [[WEIGHTS_TILE_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 5, 5, 5]
    // CHECK-SAME:      : tensor<4x5x5x5xf16> to tensor<1x5x5x5xf16>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_0]], [[WEIGHTS_TILE_0]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x1x224x224xf16>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 224, 224]
    // CHECK-SAME:      : tensor<1x10x224x224xf16> to tensor<1x5x224x224xf16>
    // CHECK:        [[WEIGHTS_TILE_1:%.+]] = VPU.Slice [[INPUT_1]] [1, 0, 0, 0] [1, 5, 5, 5]
    // CHECK-SAME:      : tensor<4x5x5x5xf16> to tensor<1x5x5x5xf16>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_1]], [[WEIGHTS_TILE_1]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x1x224x224xf16>

    // CHECK:        [[ACTIVATION_TILE_2:%.+]] = VPU.Slice [[INPUT_0]] [0, 5, 0, 0] [1, 5, 224, 224]
    // CHECK-SAME:      : tensor<1x10x224x224xf16> to tensor<1x5x224x224xf16>
    // CHECK:        [[WEIGHTS_TILE_2:%.+]] = VPU.Slice [[INPUT_1]] [2, 0, 0, 0] [1, 5, 5, 5]
    // CHECK-SAME:      : tensor<4x5x5x5xf16> to tensor<1x5x5x5xf16>

    // CHECK:        [[OUTPUT_TILE2:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_2]], [[WEIGHTS_TILE_2]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x1x224x224xf16>

    // CHECK:        [[ACTIVATION_TILE_3:%.+]] = VPU.Slice [[INPUT_0]] [0, 5, 0, 0] [1, 5, 224, 224]
    // CHECK-SAME:      : tensor<1x10x224x224xf16> to tensor<1x5x224x224xf16>
    // CHECK:        [[WEIGHTS_TILE_3:%.+]] = VPU.Slice [[INPUT_1]] [3, 0, 0, 0] [1, 5, 5, 5]
    // CHECK-SAME:      : tensor<4x5x5x5xf16> to tensor<1x5x5x5xf16>

    // CHECK:        [[OUTPUT_TILE3:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_3]], [[WEIGHTS_TILE_3]]
    // CHECK-SAME:          groups = 1 : i64
    // CHECK-SAME:          -> tensor<1x1x224x224xf16>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat(
    // CHECK-SAME:          [[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]]
    // CHECK-SAME{LITERAL}:    static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]]
    // CHECK-SAME:          -> tensor<1x4x224x224xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x4x224x224xf16>
}

// -----

// CHECK:      [[INPUT_0:%arg[0-9]]]: tensor<1x64x112x112xf16>, [[INPUT_1:%arg[0-9]]]: tensor<64x2x3x3xf16>) -> tensor<1x64x56x56xf16>
func.func @SplitSwGroupConvOverH(
        %input: tensor<1x64x112x112xf16>,
        %filter: tensor<64x2x3x3xf16>)
            -> tensor<1x64x56x56xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [0, 0],
        strides = [2, 2],
        groups = 32 : i64,
        tilingStrategy = [1, 1, 2, 1]

    } : tensor<1x64x112x112xf16>, tensor<64x2x3x3xf16> -> tensor<1x64x56x56xf16>
    return %1 : tensor<1x64x56x56xf16>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 64, 56, 112]
    // CHECK-SAME:      : tensor<1x64x112x112xf16> to tensor<1x64x56x112xf16>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_0]], [[INPUT_1]]
    // CHECK-SAME:          groups = 32 : i64
    // CHECK-SAME:          -> tensor<1x64x28x56xf16>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 55, 0] [1, 64, 57, 112]
    // CHECK-SAME:      : tensor<1x64x112x112xf16> to tensor<1x64x57x112xf16>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.GroupConvolution([[ACTIVATION_TILE_1]], [[INPUT_1]]
    // CHECK-SAME:          groups = 32 : i64
    // CHECK-SAME:          -> tensor<1x64x28x56xf16>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]]
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 28, 0]
    // CHECK-SAME:          -> tensor<1x64x56x56xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x56x56xf16>
}

// -----

// CHECK-LABEL: func.func @TilePadOverHeightWidth
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x580x580xf16>

func.func @TilePadOverHeightWidth(%arg0: tensor<1x3x580x580xf16>) -> tensor<1x16x600x600xf16> {
    %pad = VPU.Pad(%arg0)
          {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
          pads_begin_attr = [0, 0, 10, 10], pads_end_attr = [0, 13, 10, 10],
          tilingStrategy = [1, 1, 3, 3]}
          : tensor<1x3x580x580xf16> -> tensor<1x16x600x600xf16>

    return %pad : tensor<1x16x600x600xf16>

    // Tile 0

    // CHECK:        [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 190, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x190xf16>
    // CHECK:        [[PAD0:%.*]] = VPU.Pad([[SLICE0]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 10, 10], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x190x190xf16> -> tensor<1x16x200x200xf16>

    // Tile 1

    // CHECK:        [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 190] [1, 3, 190, 200]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x200xf16>
    // CHECK:        [[PAD1:%.*]] = VPU.Pad([[SLICE1]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 10, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x190x200xf16> -> tensor<1x16x200x200xf16>

    // Tile 2

    // CHECK:        [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 390] [1, 3, 190, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x190xf16>
    // CHECK:        [[PAD2:%.*]] = VPU.Pad([[SLICE2]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 10, 0], pads_end_attr = [0, 13, 0, 10]}
    // CHECK-SAME:      : tensor<1x3x190x190xf16> -> tensor<1x16x200x200xf16>

    // Tile 3

    // CHECK:        [[SLICE3:%.*]] = VPU.Slice [[INPUT]] [0, 0, 190, 0] [1, 3, 200, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x200x190xf16>
    // CHECK:        [[PAD3:%.*]] = VPU.Pad([[SLICE3]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x200x190xf16> -> tensor<1x16x200x200xf16>


    // Tile 4

    // CHECK:        [[SLICE4:%.*]] = VPU.Slice [[INPUT]] [0, 0, 190, 190] [1, 3, 200, 200]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x200x200xf16>
    // CHECK:        [[PAD4:%.*]] = VPU.Pad([[SLICE4]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x200x200xf16> -> tensor<1x16x200x200xf16>

    // Tile 5

    // CHECK:        [[SLICE5:%.*]] = VPU.Slice [[INPUT]] [0, 0, 190, 390] [1, 3, 200, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x200x190xf16>
    // CHECK:        [[PAD5:%.*]] = VPU.Pad([[SLICE5]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 10]}
    // CHECK-SAME:      : tensor<1x3x200x190xf16> -> tensor<1x16x200x200xf16>

    // Tile 6

    // CHECK:        [[SLICE6:%.*]] = VPU.Slice [[INPUT]] [0, 0, 390, 0] [1, 3, 190, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x190xf16>
    // CHECK:        [[PAD6:%.*]] = VPU.Pad([[SLICE6]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 13, 10, 0]}
    // CHECK-SAME:      : tensor<1x3x190x190xf16> -> tensor<1x16x200x200xf16>

    // Tile 7

    // CHECK:        [[SLICE7:%.*]] = VPU.Slice [[INPUT]] [0, 0, 390, 190] [1, 3, 190, 200]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x200xf16>
    // CHECK:        [[PAD7:%.*]] = VPU.Pad([[SLICE7]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 10, 0]}
    // CHECK-SAME:      : tensor<1x3x190x200xf16> -> tensor<1x16x200x200xf16>

    // Tile 8

    // CHECK:        [[SLICE8:%.*]] = VPU.Slice [[INPUT]] [0, 0, 390, 390] [1, 3, 190, 190]
    // CHECK-SAME:      : tensor<1x3x580x580xf16> to tensor<1x3x190x190xf16>
    // CHECK:        [[PAD8:%.*]] = VPU.Pad([[SLICE8]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 10, 10]}
    // CHECK-SAME:      : tensor<1x3x190x190xf16> -> tensor<1x16x200x200xf16>

    // Concat

    // CHECK:        [[CONCAT:%.*]] = VPU.Concat([[PAD0]], [[PAD1]], [[PAD2]], [[PAD3]], [[PAD4]], [[PAD5]], [[PAD6]], [[PAD7]], [[PAD8]]) {static_offsets =
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 200], [0, 0, 0, 400], [0, 0, 200, 0], [0, 0, 200, 200],
    // CHECK-SAME:      [0, 0, 200, 400], [0, 0, 400, 0], [0, 0, 400, 200], [0, 0, 400, 400]
    // CHECK-SAME:      : tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>,
    // CHECK-SAME:      tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>, tensor<1x16x200x200xf16>
    // CHECK-SAME:      -> tensor<1x16x600x600xf16>

    // CHECK:        return [[CONCAT]] : tensor<1x16x600x600xf16>
}

// -----

// CHECK-LABEL: func.func @ApplyTilingConvertSubByteInput
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x1333x1777xui4>
func.func @ApplyTilingConvertSubByteInput(%arg0: tensor<1x1x1333x1777xui4>) -> tensor<1x1x1333x1777xf16> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16, tilingStrategy = [1, 1, 3, 3]
        } : tensor<1x1x1333x1777xui4> -> tensor<1x1x1333x1777xf16>
    return %0 : tensor<1x1x1333x1777xf16>

    // CHECK:   [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 446, 594]
    // CHECK:   [[CONVERT0:%.*]] = VPU.Convert([[SLICE0]]) {dstElemType = f16}

    // CHECK:   [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 594] [1, 1, 446, 594]
    // CHECK:   [[CONVERT1:%.*]] = VPU.Convert([[SLICE1]]) {dstElemType = f16}

    // CHECK:   [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 1188] [1, 1, 446, 589]
    // CHECK:   [[CONVERT2:%.*]] = VPU.Convert([[SLICE2]]) {dstElemType = f16}

    // CHECK:   [[SLICE3:%.*]] = VPU.Slice [[INPUT]] [0, 0, 446, 0] [1, 1, 446, 594]
    // CHECK:   [[CONVERT3:%.*]] = VPU.Convert([[SLICE3]]) {dstElemType = f16}

    // CHECK:   [[SLICE4:%.*]] = VPU.Slice [[INPUT]] [0, 0, 446, 594] [1, 1, 446, 594]
    // CHECK:   [[CONVERT4:%.*]] = VPU.Convert([[SLICE4]]) {dstElemType = f16}

    // CHECK:   [[SLICE5:%.*]] = VPU.Slice [[INPUT]] [0, 0, 446, 1188] [1, 1, 446, 589]
    // CHECK:   [[CONVERT5:%.*]] = VPU.Convert([[SLICE5]]) {dstElemType = f16}

    // CHECK:   [[SLICE6:%.*]] = VPU.Slice [[INPUT]] [0, 0, 892, 0] [1, 1, 441, 594]
    // CHECK:   [[CONVERT6:%.*]] = VPU.Convert([[SLICE6]]) {dstElemType = f16}

    // CHECK:   [[SLICE7:%.*]] = VPU.Slice [[INPUT]] [0, 0, 892, 594] [1, 1, 441, 594]
    // CHECK:   [[CONVERT7:%.*]] = VPU.Convert([[SLICE7]]) {dstElemType = f16}

    // CHECK:   [[SLICE8:%.*]] = VPU.Slice [[INPUT]] [0, 0, 892, 1188] [1, 1, 441, 589]
    // CHECK:   [[CONVERT8:%.*]] = VPU.Convert([[SLICE8]]) {dstElemType = f16}

    // CHECK:   [[CONCAT:%.*]] = VPU.Concat([[CONVERT0]], [[CONVERT1]], [[CONVERT2]], [[CONVERT3]], [[CONVERT4]], [[CONVERT5]], [[CONVERT6]], [[CONVERT7]], [[CONVERT8]])

    // CHECK: return [[CONCAT]] : tensor<1x1x1333x1777xf16>
}

// -----

// CHECK-LABEL: func.func @ApplyTilingConvertNotSubByteInput
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1x1333x1777xui8>
func.func @ApplyTilingConvertNotSubByteInput(%arg0: tensor<1x1x1333x1777xui8>) -> tensor<1x1x1333x1777xf16> {
    %0 = VPU.Convert(%arg0) {
        dstElemType = f16, tilingStrategy = [1, 1, 3, 3]
        } : tensor<1x1x1333x1777xui8> -> tensor<1x1x1333x1777xf16>
    return %0 : tensor<1x1x1333x1777xf16>

    // CHECK:   [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 445, 593]
    // CHECK:   [[CONVERT0:%.*]] = VPU.Convert([[SLICE0]]) {dstElemType = f16}

    // CHECK:   [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 593] [1, 1, 445, 592]
    // CHECK:   [[CONVERT1:%.*]] = VPU.Convert([[SLICE1]]) {dstElemType = f16}

    // CHECK:   [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 1185] [1, 1, 445, 592]
    // CHECK:   [[CONVERT2:%.*]] = VPU.Convert([[SLICE2]]) {dstElemType = f16}

    // CHECK:   [[SLICE3:%.*]] = VPU.Slice [[INPUT]] [0, 0, 445, 0] [1, 1, 444, 593]
    // CHECK:   [[CONVERT3:%.*]] = VPU.Convert([[SLICE3]]) {dstElemType = f16}

    // CHECK:   [[SLICE4:%.*]] = VPU.Slice [[INPUT]] [0, 0, 445, 593] [1, 1, 444, 592]
    // CHECK:   [[CONVERT4:%.*]] = VPU.Convert([[SLICE4]]) {dstElemType = f16}

    // CHECK:   [[SLICE5:%.*]] = VPU.Slice [[INPUT]] [0, 0, 445, 1185] [1, 1, 444, 592]
    // CHECK:   [[CONVERT5:%.*]] = VPU.Convert([[SLICE5]]) {dstElemType = f16}

    // CHECK:   [[SLICE6:%.*]] = VPU.Slice [[INPUT]] [0, 0, 889, 0] [1, 1, 444, 593]
    // CHECK:   [[CONVERT6:%.*]] = VPU.Convert([[SLICE6]]) {dstElemType = f16}

    // CHECK:   [[SLICE7:%.*]] = VPU.Slice [[INPUT]] [0, 0, 889, 593] [1, 1, 444, 592]
    // CHECK:   [[CONVERT7:%.*]] = VPU.Convert([[SLICE7]]) {dstElemType = f16}

    // CHECK:   [[SLICE8:%.*]] = VPU.Slice [[INPUT]] [0, 0, 889, 1185] [1, 1, 444, 592]
    // CHECK:   [[CONVERT8:%.*]] = VPU.Convert([[SLICE8]]) {dstElemType = f16}

    // CHECK:   [[CONCAT:%.*]] = VPU.Concat([[CONVERT0]], [[CONVERT1]], [[CONVERT2]], [[CONVERT3]], [[CONVERT4]], [[CONVERT5]], [[CONVERT6]], [[CONVERT7]], [[CONVERT8]])

    // CHECK: return [[CONCAT]] : tensor<1x1x1333x1777xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u4:f16, 1.1534313725490195>
// CHECK-LABEL: func.func @ApplyTilingConvU4Weights
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @ApplyTilingConvU4Weights(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 3, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3x!qElemType, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<256x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.CastElemType<ui4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK:   [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 23, 64]
    // CHECK:   [[CONV0:%.*]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x256x22x64xf16, {order = #NHWC}>

    // CHECK:   [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 32, 23, 64]
    // CHECK:   [[CONV1:%.*]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x256x21x64xf16, {order = #NHWC}>

    // CHECK:   [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 32, 22, 64]
    // CHECK:   [[CONV2:%.*]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]]) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x256x21x64xf16, {order = #NHWC}>

    // CHECK:   [[CONCAT:%.*]] = VPU.Concat([[CONV0]], [[CONV1]], [[CONV2]])

    // CHECK: return [[CONCAT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @TileGatherDMA
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: tensor<880x960xf16>
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: tensor<1x880xsi32>
func.func @TileGatherDMA(%arg0: tensor<880x960xf16>, %arg1: tensor<1x880xsi32>) -> tensor<1x880x960xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [880, 1]} : tensor<1x880xsi32> -> tensor<880x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    %2 = VPU.GatherDMA(%arg0, %1) {
                axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]
            } : tensor<880x960xf16>, tensor<880x1xi64> -> tensor<880x960xf16>
    %3 = VPU.Reshape(%2) {shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>

    return %3 : tensor<1x880x960xf16>

    // CHECK:       [[SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [880, 1]} inputs([[INPUT_1]] : tensor<1x880xsi32>) -> tensor<880x1xsi32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[SHAPE_CAST]]) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0] [880, 480] : tensor<880x960xf16> to tensor<880x480xf16>
    // CHECK:       [[GATHER_DMA1:%.+]] = VPU.GatherDMA([[SLICE1]], [[CONVERT]]) {
    // CHECK-SAME:          axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<880x480xf16>, tensor<880x1xi64> -> tensor<880x480xf16>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 480] [880, 480] : tensor<880x960xf16> to tensor<880x480xf16>
    // CHECK:       [[GATHER_DMA2:%.+]] = VPU.GatherDMA([[SLICE2]], [[CONVERT]]) {
    // CHECK-SAME:          axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<880x480xf16>, tensor<880x1xi64> -> tensor<880x480xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER_DMA1]], [[GATHER_DMA2]]) {
    // CHECK-SAME{LITERAL}:          static_offsets = [[0, 0], [0, 480]]} : tensor<880x480xf16>, tensor<880x480xf16> -> tensor<880x960xf16>
    // CHECK:       [[AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[CONCAT]]) {
    // CHECK-SAME{LITERAL}:          dim_mapping = [[0, 1], [2]], shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>

    // CHECK:       return [[AFFINE_RESHAPE]] : tensor<1x880x960xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @ApplyTilingConvertSubByteInput
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x930x24x128xui4>
func.func @ApplyTilingConvertSubByteInput(%arg0: tensor<1x930x24x128xui4>) -> tensor<1x930x24x128xui4, {order = #NHWC}> {
    %0 = VPU.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC, tilingStrategy = [1, 2, 1, 1]} : tensor<1x930x24x128xui4> -> tensor<1x930x24x128xui4, {order = #NHWC}>
    return %0 : tensor<1x930x24x128xui4, {order = #NHWC}>

    // CHECK:   [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 480, 24, 128]
    // CHECK:   [[PERMUTE0:%.*]] = VPU.MemPermute([[SLICE0]])

    // CHECK:   [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 480, 0, 0] [1, 450, 24, 128]
    // CHECK:   [[PERMUTE1:%.*]] = VPU.MemPermute([[SLICE1]])

    // CHECK:   [[CONCAT:%.*]] = VPU.Concat([[PERMUTE0]], [[PERMUTE1]])

    // CHECK: return [[CONCAT]] : tensor<1x930x24x128xui4, {order = #NHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func.func @ApplyTilingAvgPool5D
// CHECK-SAME:          [[INPUT0:%arg[0-7]]]: tensor<1x24x8x56x56xf32>
func.func @ApplyTilingAvgPool5D(%arg0: tensor<1x24x8x56x56xf32>) -> tensor<1x24x4x56x56xf32> {
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 24, 448, 56]} : tensor<1x24x8x56x56xf32> -> tensor<1x24x448x56xf32>
    %1 = VPU.Convert(%0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x24x448x56xf32> -> tensor<1x24x448x56xf16>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 24, 8, 56, 56]} : tensor<1x24x448x56xf16> -> tensor<1x24x8x56x56xf16>
    %3 = VPU.AvgPool(%2) {exclude_pads, kernel_size = [2, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1, 1], tilingStrategy = [1, 1, 2, 1, 1]} : tensor<1x24x8x56x56xf16> -> tensor<1x24x4x56x56xf16>
    %4 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 24, 224, 56]} : tensor<1x24x4x56x56xf16> -> tensor<1x24x224x56xf16>
    %5 = VPU.Convert(%4) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x24x224x56xf16> -> tensor<1x24x224x56xf32>
    %6 = VPU.AffineReshape(%5) {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 24, 4, 56, 56]} : tensor<1x24x224x56xf32> -> tensor<1x24x4x56x56xf32>
    return %6 : tensor<1x24x4x56x56xf32>

    // CHECK:               [[INPUT_AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[INPUT0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 24, 448, 56]}
    // CHECK-SAME:              tensor<1x24x8x56x56xf32> -> tensor<1x24x448x56xf32>
    // CHECK:               [[INPUT_CONVERT:%.+]] = VPU.Convert([[INPUT_AFFINE_RESHAPE]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x24x448x56xf32> -> tensor<1x24x448x56xf16>
    // CHECK:               [[INPUT_AFFINE_RESHAPE2:%.+]] = VPU.AffineReshape([[INPUT_CONVERT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 24, 8, 56, 56]}
    // CHECK-SAME:              tensor<1x24x448x56xf16> -> tensor<1x24x8x56x56xf16>
    // CHECK:               [[SLICE_0:%.+]] = VPU.Slice [[INPUT_AFFINE_RESHAPE2]] [0, 0, 0, 0, 0] [1, 24, 4, 56, 56]
    // CHECK-SAME:              tensor<1x24x8x56x56xf16> to tensor<1x24x4x56x56xf16>
    // CHECK:               [[AVG_POOL_0:%.+]] = VPU.AvgPool([[SLICE_0]])
    // CHECK-SAME{LITERAL}:      exclude_pads, kernel_size = [2, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1, 1]
    // CHECK-SAME:               tensor<1x24x4x56x56xf16> -> tensor<1x24x2x56x56xf16>
    // CHECK:               [[SLICE_1:%.+]] = VPU.Slice [[INPUT_AFFINE_RESHAPE2]] [0, 0, 4, 0, 0] [1, 24, 4, 56, 56]
    // CHECK-SAME:              tensor<1x24x8x56x56xf16> to tensor<1x24x4x56x56xf16>
    // CHECK:               [[AVG_POOL_1:%.+]] = VPU.AvgPool([[SLICE_1]])
    // CHECK-SAME{LITERAL}:      exclude_pads, kernel_size = [2, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1, 1]
    // CHECK-SAME:               tensor<1x24x4x56x56xf16> -> tensor<1x24x2x56x56xf16>
    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[AVG_POOL_0]], [[AVG_POOL_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0, 0], [0, 0, 2, 0, 0]]}
    // CHECK-SAME:              tensor<1x24x2x56x56xf16>, tensor<1x24x2x56x56xf16> -> tensor<1x24x4x56x56xf16>
    // CHECK:               [[AVG_POOL_AFFINE_RESHAPE:%.+]] = VPU.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2], [2], [3]], shape_value = [1, 24, 224, 56]}
    // CHECK-SAME:              tensor<1x24x4x56x56xf16> -> tensor<1x24x224x56xf16>
    // CHECK:               [[AVG_POOL_CONVERT:%.+]] = VPU.Convert([[AVG_POOL_AFFINE_RESHAPE]]) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x24x224x56xf16> -> tensor<1x24x224x56xf32>
    // CHECK:               [[AVG_POOL_AFFINE_RESHAPE2:%.+]] = VPU.AffineReshape([[AVG_POOL_CONVERT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [2, 3], [4]], shape_value = [1, 24, 4, 56, 56]}
    // CHECK-SAME:              tensor<1x24x224x56xf32> -> tensor<1x24x4x56x56xf32>
    // CHECK:               return [[AVG_POOL_AFFINE_RESHAPE2]] : tensor<1x24x4x56x56xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
// CHECK-LABEL:   @DequantizeConstant
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x5x5xf16, {order = #NHWC}>
func.func @DequantizeConstant(%arg0: tensor<1x512x5x5xf16, {order = #NHWC}>) -> tensor<1x1024x5x5xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x512x1x1x!qElemType, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<1024x512x1x1xf16>, [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]
    %dequant = VPU.Dequantize(%weights) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [3, 1, 1, 1]} : tensor<1024x512x1x1x!qElemType, {order = #NHWC}> -> tensor<1024x512x1x1xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%arg0, %dequant) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [1024, 512, 1, 1], strides = [1, 1]
            } : tensor<1x512x5x5xf16, {order = #NHWC}>, tensor<1024x512x1x1xf16, {order = #NHWC}> -> tensor<1x1024x5x5xf16, {order = #NHWC}>

    return %conv : tensor<1x1024x5x5xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<352x512x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x512x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [352, 512, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<352x512x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x512x1x1xf16>, [#const.SubView<[352, 0, 0, 0], [352, 512, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<320x512x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x512x1x1xf16>, [#const.SubView<[704, 0, 0, 0], [320, 512, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK:       [[DEQUANT1:%.+]] = VPU.Dequantize([[WEIGHTS1]])
    // CHECK-SAME:              {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<352x512x1x1x!qElemType, {order = #NHWC}> -> tensor<352x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[DEQUANT2:%.+]] = VPU.Dequantize([[WEIGHTS2]])
    // CHECK-SAME:              {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<352x512x1x1x!qElemType, {order = #NHWC}> -> tensor<352x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[DEQUANT3:%.+]] = VPU.Dequantize([[WEIGHTS3]])
    // CHECK-SAME:              {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<320x512x1x1x!qElemType, {order = #NHWC}> -> tensor<320x512x1x1xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DEQUANT1]], [[DEQUANT2]], [[DEQUANT3]])

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[CONCAT]])

    // CHECK:       return [[CONV]] : tensor<1x1024x5x5xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   @TileMultiply
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x80x16x32xf16>,
// CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x80x16x32xf16>)
func.func @TileMultiply(%arg0: tensor<1x80x16x32xf16>, %arg1: tensor<1x80x16x32xf16>) -> tensor<1x80x16x32xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 2, 1, 1]} :
                tensor<1x80x16x32xf16>, tensor<1x80x16x32xf16> -> tensor<1x80x16x32xf16>

    return %0 : tensor<1x80x16x32xf16>

    // CHECK:   [[SLICE00:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 48, 16, 32]
    // CHECK:   [[SLICE01:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 48, 16, 32]
    // CHECK:   [[MULTIPLY0:%.+]] = VPU.Multiply([[SLICE00]], [[SLICE01]])

    // CHECK:   [[SLICE10:%.+]] = VPU.Slice [[INPUT0]] [0, 48, 0, 0] [1, 32, 16, 32]
    // CHECK:   [[SLICE11:%.+]] = VPU.Slice [[INPUT1]] [0, 48, 0, 0] [1, 32, 16, 32]
    // CHECK:   [[MULTIPLY1:%.+]] = VPU.Multiply([[SLICE10]], [[SLICE11]])

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[MULTIPLY0]], [[MULTIPLY1]])

    // CHECK: return [[CONCAT]] : tensor<1x80x16x32xf16>
}

// -----

// CHECK-LABEL:   @TileMultiplyNotAlign
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x31x16x32xf16>,
// CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x31x16x32xf16>)
func.func @TileMultiplyNotAlign(%arg0: tensor<1x31x16x32xf16>, %arg1: tensor<1x31x16x32xf16>) -> tensor<1x31x16x32xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 2, 1, 1]} :
                tensor<1x31x16x32xf16>, tensor<1x31x16x32xf16> -> tensor<1x31x16x32xf16>

    return %0 : tensor<1x31x16x32xf16>

    // CHECK:   [[SLICE00:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 16, 16, 32]
    // CHECK:   [[SLICE01:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 16, 16, 32]
    // CHECK:   [[MULTIPLY0:%.+]] = VPU.Multiply([[SLICE00]], [[SLICE01]])

    // CHECK:   [[SLICE10:%.+]] = VPU.Slice [[INPUT0]] [0, 16, 0, 0] [1, 15, 16, 32]
    // CHECK:   [[SLICE11:%.+]] = VPU.Slice [[INPUT1]] [0, 16, 0, 0] [1, 15, 16, 32]
    // CHECK:   [[MULTIPLY1:%.+]] = VPU.Multiply([[SLICE10]], [[SLICE11]])

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[MULTIPLY0]], [[MULTIPLY1]])

    // CHECK: return [[CONCAT]] : tensor<1x31x16x32xf16>
}

// -----

// CHECK-LABEL: @ApplyTilingReverse
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf16>
func.func @ApplyTilingReverse(%arg0: tensor<512x512x3x3xf16>) -> tensor<512x512x3x3xf16> {
    %0 = VPU.Reverse(%arg0) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>, tilingStrategy = [7, 1, 1, 1]} : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>
    return %0 : tensor<512x512x3x3xf16>

    // CHECK:   [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [74, 512, 3, 3]
    // CHECK:   [[REVERSE0:%.+]] = VPU.Reverse([[SLICE0]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [74, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE1:%.+]] = VPU.Reverse([[SLICE1]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [147, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE2:%.+]] = VPU.Reverse([[SLICE2]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [220, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE3:%.+]] = VPU.Reverse([[SLICE3]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE4:%.+]] = VPU.Slice [[INPUT]] [293, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE4:%.+]] = VPU.Reverse([[SLICE4]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE5:%.+]] = VPU.Slice [[INPUT]] [366, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE5:%.+]] = VPU.Reverse([[SLICE5]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[SLICE6:%.+]] = VPU.Slice [[INPUT]] [439, 0, 0, 0] [73, 512, 3, 3]
    // CHECK:   [[REVERSE6:%.+]] = VPU.Reverse([[SLICE6]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>}

    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[REVERSE0]], [[REVERSE1]], [[REVERSE2]], [[REVERSE3]], [[REVERSE4]], [[REVERSE5]], [[REVERSE6]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [74, 0, 0, 0], [147, 0, 0, 0], [220, 0, 0, 0], [293, 0, 0, 0], [366, 0, 0, 0], [439, 0, 0, 0]]}

    // CHECK: return [[CONCAT]] : tensor<512x512x3x3xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWithSEPNoMC
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x192x1x1xf16, {order = #NHWC}>)
func.func @DWConvWithSEPNoMC(%arg0: tensor<1x192x1x1xf16, {order = #NHWC}>) -> tensor<1x192x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<192x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<192x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x192x2x2xi1> = dense<1> : tensor<1x192x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16, seDepth = 18,
        seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 192, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 192, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 192, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x192x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x192x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 192, 2, 2]>>

    %dwconv = VPU.NCE.DepthConvolution(%input, %weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [192, 1, 1, 1],
        strides = [1, 1],
        tilingStrategy = [1, 4, 1, 1]
    } -> tensor<1x192x2x2xf16, {order = #NHWC}>

    return %dwconv : tensor<1x192x2x2xf16, {order = #NHWC}>

    // To satisfy DW.Conv + SEP requirements for workload channels, the division is:
    // [64, 64, 32, 32]

    // CHECK:     [[SE0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:   dataShape = [1, 64, 1, 1],
    // CHECK-SAME:   offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]
    // CHECK-SAME:   seDepth = 4 : i64, seSize = [16, 16, 16, 16]
    // CHECK-SAME:  -> tensor<1x4x2x2xi32, {order = #NHWC}>

    // CHECK:     [[SE1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:   dataShape = [1, 64, 1, 1],
    // CHECK-SAME:   offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]
    // CHECK-SAME:   seDepth = 4 : i64, seSize = [16, 16, 16, 16]
    // CHECK-SAME:  -> tensor<1x4x2x2xi32, {order = #NHWC}>

    // CHECK:     [[SE2:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:   dataShape = [1, 32, 1, 1],
    // CHECK-SAME:   offsets = [0, 0, 0, 0], sizes = [1, 32, 2, 2]
    // CHECK-SAME:   seDepth = 2 : i64, seSize = [16, 16]
    // CHECK-SAME:  -> tensor<1x2x2x2xi32, {order = #NHWC}>

    // CHECK:     [[SE3:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:   dataShape = [1, 32, 1, 1],
    // CHECK-SAME:   offsets = [0, 0, 0, 0], sizes = [1, 32, 2, 2]
    // CHECK-SAME:   seDepth = 2 : i64, seSize = [16, 16]
    // CHECK-SAME:  -> tensor<1x2x2x2xi32, {order = #NHWC}>

    // CHECK:      [[DATA3:%.+]] = VPU.Slice [[ARG0]] [0, 160, 0, 0] [1, 32, 1, 1]
    // CHECK-SAME:     tensor<1x192x1x1xf16, {order = #NHWC}> to tensor<1x32x1x1xf16, {order = #NHWC}>
    // CHECK:      [[IN3:%.+]] = VPU.GroupSparseTensor([[DATA3]], {{[^:]+}}, [[SE3]])

    // CHECK:      [[DATA2:%.+]] = VPU.Slice [[ARG0]] [0, 128, 0, 0] [1, 32, 1, 1]
    // CHECK-SAME:     tensor<1x192x1x1xf16, {order = #NHWC}> to tensor<1x32x1x1xf16, {order = #NHWC}>
    // CHECK:      [[IN2:%.+]] = VPU.GroupSparseTensor([[DATA2]], {{[^:]+}}, [[SE2]])

    // CHECK:      [[DATA1:%.+]] = VPU.Slice [[ARG0]] [0, 64, 0, 0] [1, 64, 1, 1]
    // CHECK-SAME:     tensor<1x192x1x1xf16, {order = #NHWC}> to tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:      [[IN1:%.+]] = VPU.GroupSparseTensor([[DATA1]], {{[^:]+}}, [[SE1]])

    // CHECK:      [[DATA0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 64, 1, 1]
    // CHECK-SAME:     tensor<1x192x1x1xf16, {order = #NHWC}> to tensor<1x64x1x1xf16, {order = #NHWC}>
    // CHECK:      [[IN0:%.+]] = VPU.GroupSparseTensor([[DATA0]], {{[^:]+}}, [[SE0]])

    // CHECK: [[OUT0:%.+]] = VPU.NCE.DepthConvolution([[IN0]]
    // CHECK-SAME:  -> tensor<1x64x2x2xf16, {order = #NHWC}>
    // CHECK: [[OUT1:%.+]] = VPU.NCE.DepthConvolution([[IN1]]
    // CHECK-SAME:  -> tensor<1x64x2x2xf16, {order = #NHWC}>
    // CHECK: [[OUT2:%.+]] = VPU.NCE.DepthConvolution([[IN2]]
    // CHECK-SAME:  -> tensor<1x32x2x2xf16, {order = #NHWC}>
    // CHECK: [[OUT3:%.+]] = VPU.NCE.DepthConvolution([[IN3]]
    // CHECK-SAME:  -> tensor<1x32x2x2xf16, {order = #NHWC}>

    // CHECK:              VPU.Concat([[OUT0]], [[OUT1]], [[OUT2]], [[OUT3]]) {
    // CHECK-SAME{LITERAL}:   static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 128, 0, 0], [0, 160, 0, 0]]}
    // CHECK-SAME:          tensor<1x64x2x2xf16, {order = #NHWC}>, tensor<1x64x2x2xf16, {order = #NHWC}>, tensor<1x32x2x2xf16, {order = #NHWC}>, tensor<1x32x2x2xf16, {order = #NHWC}> -> tensor<1x192x2x2xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   @ApplyTilingSubtractWithDifferentInnerDimSize
// CHECK-SAME:          [[INPUT0:%arg[0-9]]]: tensor<1x256x1x7000xf16>
// CHECK-SAME:          [[INPUT1:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @ApplyTilingSubtractWithDifferentInnerDimSize(%arg0: tensor<1x256x1x7000xf16>, %arg1: tensor<1x256x1x1xf16>) -> tensor<1x256x1x7000xf16> {
    %0 = VPU.Subtract(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        tilingStrategy = [1, 1, 1, 3]
    } : tensor<1x256x1x7000xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x7000xf16>

    return %0 : tensor<1x256x1x7000xf16>

    // CHECK:      [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 256, 1, 2336]
    // CHECK-SAME:   : tensor<1x256x1x7000xf16> to tensor<1x256x1x2336xf16>
    // CHECK:      [[SUBTRACT0:%.+]] = VPU.Subtract([[SLICE0]], [[INPUT1]])

    // CHECK:      [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 2336] [1, 256, 1, 2336]
    // CHECK-SAME:   : tensor<1x256x1x7000xf16> to tensor<1x256x1x2336xf16>
    // CHECK:      [[SUBTRACT1:%.+]] = VPU.Subtract([[SLICE1]], [[INPUT1]])

    // CHECK:      [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 4672] [1, 256, 1, 2328]
    // CHECK-SAME:   : tensor<1x256x1x7000xf16> to tensor<1x256x1x2328xf16>
    // CHECK:      [[SUBTRACT2:%.+]] = VPU.Subtract([[SLICE2]], [[INPUT1]])

    // CHECK:      [[OUTPUT:%.+]] = VPU.Concat([[SUBTRACT0]], [[SUBTRACT1]], [[SUBTRACT2]])
    // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 2336], [0, 0, 0, 4672]]}
    // CHECK-SAME:   : tensor<1x256x1x2336xf16>, tensor<1x256x1x2336xf16>, tensor<1x256x1x2328xf16> -> tensor<1x256x1x7000xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x1x7000xf16>
}
