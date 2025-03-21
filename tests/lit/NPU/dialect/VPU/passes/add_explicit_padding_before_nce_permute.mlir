//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file  --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --add-explicit-padding-before-nce-permute %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @InsertExplicitPadding
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x166x1x16xf16>)
func.func @InsertExplicitPadding(%arg0:  tensor<1x166x1x16xf16>) -> tensor<1x160x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x176x1x3xf16, {order = #NHWC}>
        = dense<1.000000e+00> : tensor<160x166x3xf16>, [
            #const.Reshape<[160, 166, 1, 3]>,
            #const.Reorder<#NHWC>,
            #const.PadWithZero<[0, 0, 0, 0], [0, 10, 0, 0]>
        ]

    %weights_table = const.Declare tensor<160x1x1x4xsi32> = dense<1> : tensor<160x1x1x4xsi32>

    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 176 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x176x1x16xf16, {order = #NHWC}>

    %slice = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 176, 1, 3] :
        tensor<1x176x1x16xf16, {order = #NHWC}>
        to tensor<1x176x1x3xf16, {order = #NHWC}>


    %conv = VPU.NCE.Convolution(%slice, %weights, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [160, 176, 1, 3], strides = [1, 1]}
        -> tensor<1x160x1x1xf16, {order = #NHWC}>


    return %conv : tensor<1x160x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<160x176x1x3xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32>
    // CHECK-DAG:       [[ZERO_CONST:%.+]] = const.Declare tensor<1x10x1x16xf16> = dense<0.000000e+00> : tensor<1x10x1x16xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INPUT]], [[ZERO_CONST]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 1 : i64>} :
    // CHECK-SAME:      tensor<1x166x1x16xf16>, tensor<1x10x1x16xf16> -> tensor<1x176x1x16xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]]  = VPU.NCE.Permute([[CONCAT]])
    // CHECK-SAME:      dstElemType = f16, dstOrder = #NHWC,
    // CHECK-SAME:      expandedChannels = 176 : i64,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>}
    // CHECK-SAME:  -> tensor<1x176x1x16xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE_PERMUTE]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 176, 1, 3]
    // CHECK-SAME:      : tensor<1x176x1x16xf16, {order = #NHWC}>
    // CHECK-SAME:  to tensor<1x176x1x3xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [160, 176, 1, 3], strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x160x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x160x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @NCEPermuteDepthConvWeightsSparseConv
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x257x257xf16>)
func.func @NCEPermuteDepthConvWeightsSparseConv(%arg0: tensor<1x3x257x257xf16>) ->  tensor<1x16x129x129xf16, {order = #NHWC}> {
    %conv_weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %depth_conv_weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %depth_conv_weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> :
            tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 3 : i64>, #const.Reorder<#NHWC>,
            #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>, #const.Reorder<#NCHW>, #const.Reshape<[16, 1, 1, 1]>,
            #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %conv_weights = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %conv_weights_sparsity_map = const.Declare tensor<16x1x1x256xi1> = dense<1.0> : tensor<16x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

    %weights_sparse_group = VPU.GroupSparseTensor(%conv_weights, %conv_weights_sparsity_map) {
        sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<16xi64>,
        alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>,
            sparsity_map=tensor<16x1x1x256xi1>, is_weights,
            #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<16xi64>, alignment = 16 : i64>>


    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
        : tensor<1x3x257x257xf16> -> tensor<1x3x257x272xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x16x257x272xf16, {order = #NHWC}>

    %slice = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 16, 257, 257]
        : tensor<1x16x257x272xf16, {order = #NHWC}> to tensor<1x16x257x257xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%slice, %depth_conv_weights, %depth_conv_weights_table)
        {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [16, 1, 1, 1], strides = [1, 1]}
        -> tensor<1x16x257x257xf16, {order = #NHWC}>

    %conv = VPU.NCE.Convolution(%depth_conv, %weights_sparse_group, %conv_weights_table)
        {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [16, 16, 3, 3], strides = [2, 2]}
        -> tensor<1x16x129x129xf16, {order = #NHWC}>

    return %conv : tensor<1x16x129x129xf16, {order = #NHWC}>

    // CHECK:       [[WT_CONV:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:       [[WT_DEPTHCONV:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:       [[WEIGHTS_DEPTHCONV:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_CONV:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_SM_CONV:%.+]] = const.Declare tensor<16x1x1x256xi1>

    // CHECK:       [[SPARSE_GROUP:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_CONV]], [[WEIGHTS_SM_CONV]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<16xi64>,
    // CHECK-SAME:      alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<16x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<16x1x1x256xi1>,
    // CHECK-SAME:      is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<16xi64>, alignment = 16 : i64>>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 15]}
    // CHECK-SAME:      : tensor<1x3x257x257xf16> -> tensor<1x3x257x272xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x257x272xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 16, 257, 257]
    // CHECK-SAME:      : tensor<1x16x257x272xf16, {order = #NHWC}> to tensor<1x16x257x257xf16, {order = #NHWC}>

    // CHECK:       [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE]], [[WEIGHTS_DEPTHCONV]], [[WT_DEPTHCONV]])
    // CHECK-SAME:      -> tensor<1x16x257x257xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[DEPTH_CONV]], [[SPARSE_GROUP]], [[WT_CONV]])
    // CHECK-SAME:      -> tensor<1x16x129x129xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x16x129x129xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @NCEPermuteMultiAvgPoolSliceOnChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x77x33x21xf16>)
func.func @NCEPermuteMultiAvgPoolSliceOnChannels(%arg0: tensor<1x77x33x21xf16>) ->  tensor<1x77x1x1xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 11]}
        : tensor<1x77x33x21xf16> -> tensor<1x77x33x32xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 80 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x80x33x32xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 80, 33, 21]
        : tensor<1x80x33x32xf16, {order = #NHWC}> to tensor<1x80x33x21xf16, {order = #NHWC}>

    %avg_pool0 = VPU.NCE.AveragePool(%slice0)
        {kernel_size = [3, 7],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [3, 7]}
        -> tensor<1x80x11x3xf16, {order = #NHWC}>

    %avg_pool1 = VPU.NCE.AveragePool(%avg_pool0)
        {kernel_size = [11, 3],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]}
        -> tensor<1x80x1x1xf16, {order = #NHWC}>

    %slice1 = VPU.Slice %avg_pool1 [0, 0, 0, 0] [1, 77, 1, 1]
        : tensor<1x80x1x1xf16, {order = #NHWC}> to tensor<1x77x1x1xf16, {order = #NHWC}>

    return %slice1 : tensor<1x77x1x1xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 11]}
    // CHECK-SAME:      : tensor<1x77x33x21xf16> -> tensor<1x77x33x32xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 80 : i64, ppe = #VPU.PPEStub<>
    // CHECK-SAME:      -> tensor<1x80x33x32xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 80, 33, 21]
    // CHECK-SAME:      : tensor<1x80x33x32xf16, {order = #NHWC}> to tensor<1x80x33x21xf16, {order = #NHWC}>

    // CHECK:       [[AVG_POOL0:%.+]] = VPU.NCE.AveragePool([[SLICE0]])
    // CHECK-SAME:      -> tensor<1x80x11x3xf16, {order = #NHWC}>

    // CHECK:       [[AVG_POOL1:%.+]] = VPU.NCE.AveragePool([[AVG_POOL0]])
    // CHECK-SAME:      -> tensor<1x80x1x1xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[AVG_POOL1]] [0, 0, 0, 0] [1, 77, 1, 1]
    // CHECK-SAME:      : tensor<1x80x1x1xf16, {order = #NHWC}> to tensor<1x77x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x77x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @NCEPermuteSingleAvgPoolSliceOnChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x77x11x7xf16>)
func.func @NCEPermuteSingleAvgPoolSliceOnChannels(%arg0: tensor<1x77x11x7xf16>) ->  tensor<1x77x1x1xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9]}
        : tensor<1x77x11x7xf16> -> tensor<1x77x11x16xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 80 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x80x11x16xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 80, 11, 7]
        : tensor<1x80x11x16xf16, {order = #NHWC}> to tensor<1x80x11x7xf16, {order = #NHWC}>

    %avg_pool = VPU.NCE.AveragePool(%slice0)
        {kernel_size = [11, 7],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]}
        -> tensor<1x80x1x1xf16, {order = #NHWC}>

    %slice1 = VPU.Slice %avg_pool [0, 0, 0, 0] [1, 77, 1, 1]
        : tensor<1x80x1x1xf16, {order = #NHWC}> to tensor<1x77x1x1xf16, {order = #NHWC}>

    return %slice1 : tensor<1x77x1x1xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 9]}
    // CHECK-SAME:      : tensor<1x77x11x7xf16> -> tensor<1x77x11x16xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 80 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x80x11x16xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 80, 11, 7]
    // CHECK-SAME:      : tensor<1x80x11x16xf16, {order = #NHWC}> to tensor<1x80x11x7xf16, {order = #NHWC}>

    // CHECK:       [[AVG_POOL:%.+]] = VPU.NCE.AveragePool([[SLICE0]])
    // CHECK-SAME:      -> tensor<1x80x1x1xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[AVG_POOL]] [0, 0, 0, 0] [1, 77, 1, 1]
    // CHECK-SAME:      : tensor<1x80x1x1xf16, {order = #NHWC}> to tensor<1x77x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x77x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @NCEPermuteDepthConvSliceOnChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x5929x3x3xf16>)
func.func @NCEPermuteDepthConvSliceOnChannels(%arg0: tensor<1x5929x3x3xf16>) ->  tensor<1x5929x3x3xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5936x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf32>,
        [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 77 : i64>, #const.Broadcast<0 : i64, 77 : i64>,
        #const.Reshape<[1, 5929, 1, 1]>, #const.Reshape<[5929, 1, 1, 1]>, #const.Reorder<#NHWC>,
        #const.PadWithZero<[0, 0, 0, 0], [7, 0, 0, 0]>, #const.Reorder<#NCHW>, #const.Reshape<[5936, 1, 1, 1]>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<5936x1x1x4xsi32> = dense<1> : tensor<5936x1x1x4xsi32>


    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]}
        : tensor<1x5929x3x3xf16> -> tensor<1x5929x3x16xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 5936 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x5936x3x16xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 5936, 3, 3]
        : tensor<1x5936x3x16xf16, {order = #NHWC}> to tensor<1x5936x3x3xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%slice0, %weights, %weights_table)
        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [5936, 1, 1, 1], strides = [1, 1]}
        -> tensor<1x5936x3x3xf16, {order = #NHWC}>

    %slice1 = VPU.Slice %depth_conv [0, 0, 0, 0] [1, 5929, 3, 3]
        : tensor<1x5936x3x3xf16, {order = #NHWC}> to tensor<1x5929x3x3xf16, {order = #NHWC}>

    return %slice1 : tensor<1x5929x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<5936x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WT:%.+]] = const.Declare tensor<5936x1x1x4xsi32> = dense<1> : tensor<5936x1x1x4xsi32>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]}
    // CHECK-SAME:      : tensor<1x5929x3x3xf16> -> tensor<1x5929x3x16xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 5936 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x5936x3x16xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 5936, 3, 3]
    // CHECK-SAME:      : tensor<1x5936x3x16xf16, {order = #NHWC}> to tensor<1x5936x3x3xf16, {order = #NHWC}>

    // CHECK:       [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      -> tensor<1x5936x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[DEPTH_CONV]] [0, 0, 0, 0] [1, 5929, 3, 3]
    // CHECK-SAME:      : tensor<1x5936x3x3xf16, {order = #NHWC}> to tensor<1x5929x3x3xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x5929x3x3xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @NCEPermuteDepthConvAvgPoolSliceOnChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x39x13x3xf16>)
func.func @NCEPermuteDepthConvAvgPoolSliceOnChannels(%arg0: tensor<1x39x13x3xf16>) ->  tensor<1x39x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<48x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<39x1x7x3xf16>,
        [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [9, 0, 0, 0]>,
        #const.Reorder<#NCHW>, #const.Reshape<[48, 21, 1, 1]>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]}
        : tensor<1x39x13x3xf16> -> tensor<1x39x13x16xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x48x13x16xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 48, 13, 3]
        : tensor<1x48x13x16xf16, {order = #NHWC}> to tensor<1x48x13x3xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%slice0, %weights, %weights_table)
        {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [48, 1, 7, 3], strides = [7, 1]}
        -> tensor<1x48x2x1xf16, {order = #NHWC}>

    %avg_pool = VPU.NCE.AveragePool(%depth_conv)
        {kernel_size = [2, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]}
        -> tensor<1x48x1x1xf16, {order = #NHWC}>

    %slice1 = VPU.Slice %avg_pool [0, 0, 0, 0] [1, 39, 1, 1]
        : tensor<1x48x1x1xf16, {order = #NHWC}> to tensor<1x39x1x1xf16, {order = #NHWC}>

    return %slice1 : tensor<1x39x1x1xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<48x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<39x1x7x3xf16>
    // CHECK:       [[WT:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 13]}
    // CHECK-SAME:      : tensor<1x39x13x3xf16> -> tensor<1x39x13x16xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x48x13x16xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 48, 13, 3]
    // CHECK-SAME:      : tensor<1x48x13x16xf16, {order = #NHWC}> to tensor<1x48x13x3xf16, {order = #NHWC}>

    // CHECK:       [[DEPTH_CONV:%.+]] = VPU.NCE.DepthConvolution([[SLICE0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      -> tensor<1x48x2x1xf16, {order = #NHWC}>

    // CHECK:       [[AVG_POOL:%.+]] = VPU.NCE.AveragePool([[DEPTH_CONV]])
    // CHECK-SAME:      -> tensor<1x48x1x1xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[AVG_POOL]] [0, 0, 0, 0] [1, 39, 1, 1]
    // CHECK-SAME:      : tensor<1x48x1x1xf16, {order = #NHWC}> to tensor<1x39x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x39x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:  @NCEPermuteEltwiseSliceOnChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x77x77xf16>)
func.func @NCEPermuteEltwiseSliceOnChannels(%arg0: tensor<1x12x77x77xf16>) ->  tensor<1x12x77x77xf16, {order = #NHWC}> {
    %constant = const.Declare tensor<1x16x77x77xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x77x77xf32>,
        [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 12 : i64>, #const.Reorder<#NHWC>,
        #const.PadWithZero<[0, 0, 0, 0], [0, 4, 0, 0]>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 3]}
        : tensor<1x12x77x77xf16> -> tensor<1x12x77x80xf16>

    %nce_permute = VPU.NCE.Permute(%expand)
        {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x16x77x80xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %nce_permute [0, 0, 0, 0] [1, 16, 77, 77]
        : tensor<1x16x77x80xf16, {order = #NHWC}> to tensor<1x16x77x77xf16, {order = #NHWC}>

    %eltwise = VPU.NCE.Eltwise(%slice0, %constant) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        -> tensor<1x16x77x77xf16, {order = #NHWC}>

    %slice1 = VPU.Slice %eltwise [0, 0, 0, 0] [1, 12, 77, 77]
        : tensor<1x16x77x77xf16, {order = #NHWC}> to tensor<1x12x77x77xf16, {order = #NHWC}>

    return %slice1 : tensor<1x12x77x77xf16, {order = #NHWC}>


    // CHECK:       [[CONST:%.+]] = const.Declare tensor<1x16x77x77xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x77x77xf32>

    // CHECK:       [[EXPAND:%.+]] = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 3]}
    // CHECK-SAME:      : tensor<1x12x77x77xf16> -> tensor<1x12x77x80xf16>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[EXPAND]])
    // CHECK-SAME:      {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x77x80xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[NCE_PERMUTE]] [0, 0, 0, 0] [1, 16, 77, 77]
    // CHECK-SAME:      : tensor<1x16x77x80xf16, {order = #NHWC}> to tensor<1x16x77x77xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]]  = VPU.NCE.Eltwise([[SLICE0]], [[CONST]])
    // CHECK-SAME:      -> tensor<1x16x77x77xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[ELTWISE]] [0, 0, 0, 0] [1, 12, 77, 77]
    // CHECK-SAME:      : tensor<1x16x77x77xf16, {order = #NHWC}> to tensor<1x12x77x77xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE1]] : tensor<1x12x77x77xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8<0:254>:f16, 0.0078740157480314959:127>

// CHECK-LABEL:  @QuantizedNCEPermuteCompressConv
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x64x64xf16, {order = #NCHW}>)
func.func @QuantizedNCEPermuteCompressConv(%arg0: tensor<1x4x64x64xf16, {order = #NCHW}>) ->  tensor<1x16x32x32x!qElemType, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x48xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %nce_permute = VPU.NCE.Permute(%arg0)
        {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
        -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    %conv = VPU.NCE.CompressConvolution(%nce_permute, %weights, %weights_table)
        {cm_sp_pattern = 1 : i64,
        pad = #VPU.Padding<left = 0 : i64,
                            right = 1 : i64,
                            top = 0 : i64,
                            bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
        -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    return %conv : tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x48x!qElemType, {order = #NHWC}>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x4x64x64x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.CompressConvolution([[NCE_PERMUTE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      {cm_sp_pattern = 1 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [16, 1, 3, 3], strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x16x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x16x32x32x!qElemType, {order = #NHWC}>
}
