//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --apply-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @SplitNCEPermute(%arg0: tensor<1x31x224x224xf16>) -> tensor<1x32x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 32 : i64,
        ppe = #VPU.PPEStub<>,
        tilingStrategy = [1, 2, 1, 1]
    } -> tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 224, 224]
    // CHECK-SAME:      : tensor<1x31x224x224xf16> to tensor<1x16x224x224xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Permute([[INPUT_TILE0]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 16 : i64,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice %arg0 [0, 16, 0, 0] [1, 15, 224, 224]
    // CHECK-SAME:      : tensor<1x31x224x224xf16> to tensor<1x15x224x224xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Permute([[INPUT_TILE1]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 16 : i64,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x16x224x224x!qElemType, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 16, 0, 0]
    // CHECK-SAME:      -> tensor<1x32x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x224x224x!qElemType, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ApplyTilingSETransposedConv
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @ApplyTilingSETransposedConv(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x382x382xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1x16x386x386xi1, {order = #NHWC}> = dense<1> : tensor<1x16x386x386xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_1 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>
    %0 = VPU.StorageElementTable {
            dataElemType = f16, dataShape = [1, 16, 128, 128], seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>, seDepth = 1 : i64, seSize = 16 : i64
        } -> tensor<1x1x386x386xi32, {order = #NHWC}>

    %1 = VPU.GroupSparseTensor(%arg0, %cst_0, %0) {
            seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>
        } -> !VPU.SparseTensor<data=tensor<1x16x128x128xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x16x386x386xi1, {order = #NHWC}>,
                               storage_element_table=tensor<1x1x386x386xi32, {order = #NHWC}>,
            #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 1, 3]>>

    %2 = VPU.NCE.Convolution(%1, %cst_1, %cst) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [16, 16, 5, 5], strides = [1, 1],
            tilingStrategy = [1, 1, 1, 2]
        } -> tensor<1x16x382x382xf16, {order = #NHWC}>

    return %2 : tensor<1x16x382x382xf16, {order = #NHWC}>

    // CHECK-DAG:   [[INPUT_SM_0:%.+]] = const.Declare tensor<1x16x386x195xi1, {order = #NHWC}> =
    // CHECK-SAME:          : tensor<1x16x386x386xi8>, [#const.SubView<[0, 0, 0, 0], [1, 16, 386, 195]>, #const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK-DAG:   [[INPUT_SM_1:%.+]] = const.Declare tensor<1x16x386x195xi1, {order = #NHWC}> =
    // CHECK-SAME:          : tensor<1x16x386x386xi8>, [#const.SubView<[0, 0, 0, 191], [1, 16, 386, 195]>, #const.Reorder<#NHWC>, #const.CastElemType<i1>]

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_SE_0:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 64],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 3, 5],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 16, 386, 195]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64} -> tensor<1x1x386x195xi32, {order = #NHWC}>
    // CHECK:       [[INPUT_SE_1:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 16, 128, 66],
    // CHECK-SAME:          seAttr = #VPU.SEUpsampling<factors = [2, 2], padding = [3, 1, 3, 5],
    // CHECK-SAME:          offsets = [0, 0, 0, 5], sizes = [1, 16, 386, 195]>,
    // CHECK-SAME:          seDepth = 1 : i64, seSize = 16 : i64} -> tensor<1x1x386x195xi32, {order = #NHWC}>

    // CHECK:       [[INPUT_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 62] [1, 16, 128, 66]
    // CHECK:       [[INPUT_SPARSE_1:%.+]] = VPU.GroupSparseTensor([[INPUT_1]], [[INPUT_SM_1]], [[INPUT_SE_1]])
    // CHECK:       [[INPUT_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 128, 64]
    // CHECK:       [[INPUT_SPARSE_0:%.+]] = VPU.GroupSparseTensor([[INPUT_0]], [[INPUT_SM_0]], [[INPUT_SE_0]])

    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE_1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 191]]} : tensor<1x16x382x191xf16, {order = #NHWC}>, tensor<1x16x382x191xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x16x382x382xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSEDepthConvolutionAlignedOnChannels
// CHECK-SAME:      [[ARG:%arg[0-9]]]:  tensor<1x1024x65x65xf16, {order = #NHWC}>
func.func @SplitSEDepthConvolutionAlignedOnChannels(%arg0: tensor<1x1024x65x65xf16, {order = #NHWC}>) -> tensor<1x1024x32x32xf16> {
    %sparsity_map = const.Declare tensor<1x1024x32x32xi1, {order = #NHWC}> = dense<1> : tensor<1x1024x32x32xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %weight_table = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %weight = const.Declare tensor<1024x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1x1x3x3xf32>,
            [#const.Reshape<[1024, 1, 3, 3]>, #const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[1024, 9, 1, 1]>,
            #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %set = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 1024, 65, 65],
        seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1], dataSizes = [1, 1024, 64, 64]>,
        seDepth = 16 : i64, seSize = 64 : i64} -> tensor<1x16x32x32xi32, {order = #NHWC}>
    %8 = VPU.GroupSparseTensor(%arg0, %sparsity_map, %set) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 1, 1],
        dataSizes = [1, 1024, 64, 64]>} -> !VPU.SparseTensor<data=tensor<1x1024x65x65xf16, {order =#NHWC}>,
        sparsity_map=tensor<1x1024x32x32xi1, {order =#NHWC}>, storage_element_table=tensor<1x16x32x32xi32,
        {order =#NHWC}>, #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3],
        dataOffset = [0, 0, 1, 1], dataSizes = [1, 1024, 64, 64]>>
    %9 = VPU.NCE.DepthConvolution(%8, %weight, %weight_table) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64,
        right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [1024, 1, 3, 3], strides = [1, 1], tilingStrategy = [1, 3, 1, 1]} -> tensor<1x1024x32x32xf16>

    return %9 : tensor<1x1024x32x32xf16>

    // CHECK:       [[SPARSE_SLICE0:%.+]] = VPU.Slice [[ARG]]
    // CHECK-SAME:      [0, 768, 0, 0] [1, 256, 65, 65] : tensor<1x1024x65x65xf16, {order = #NHWC}> to tensor<1x256x65x65xf16, {order = #NHWC}>
    // CHECK:       [[SPARSE_TENSOR0:%.+]] = VPU.GroupSparseTensor([[SPARSE_SLICE0]],
    // CHECK-SAME:      dataSizes = [1, 256, 63, 63], offsets = [], sizes = [1, 256, 32, 32]>} ->
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x256x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x32x32xi1

    // CHECK:       [[SPARSE_SLICE1:%.+]] = VPU.Slice [[ARG]]
    // CHECK-SAME:      [0, 384, 0, 0] [1, 384, 65, 65] : tensor<1x1024x65x65xf16, {order = #NHWC}> to tensor<1x384x65x65xf16, {order = #NHWC}>
    // CHECK:       [[SPARSE_TENSOR1:%.+]] = VPU.GroupSparseTensor([[SPARSE_SLICE1]],
    // CHECK-SAME:      dataSizes = [1, 384, 63, 63], offsets = [], sizes = [1, 384, 32, 32]>} ->
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x384x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x32x32xi1

    // CHECK:       [[SPARSE_SLICE2:%.+]] = VPU.Slice [[ARG]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 384, 65, 65] : tensor<1x1024x65x65xf16, {order = #NHWC}> to tensor<1x384x65x65xf16, {order = #NHWC}>
    // CHECK:       [[SPARSE_TENSOR2:%.+]] = VPU.GroupSparseTensor([[SPARSE_SLICE2]],
    // CHECK-SAME:      dataSizes = [1, 384, 63, 63], offsets = [], sizes = [1, 384, 32, 32]>} ->
    // CHECK-SAME:      !VPU.SparseTensor<data=tensor<1x384x65x65xf16, {order = #NHWC}>, sparsity_map=tensor<1x384x32x32xi1

    // CHECK:       [[DCONV0:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_TENSOR2]]
    // CHECK-SAME:      -> tensor<1x384x32x32xf16>
    // CHECK:       [[DCONV2:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_TENSOR1]]
    // CHECK-SAME:      -> tensor<1x384x32x32xf16>
    // CHECK:       [[DCONV3:%.+]] = VPU.NCE.DepthConvolution([[SPARSE_TENSOR0]]
    // CHECK-SAME:      -> tensor<1x256x32x32xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[DCONV0]], [[DCONV2]], [[DCONV3]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 384, 0, 0], [0, 768, 0, 0]]}
    // CHECK-SAME:           -> tensor<1x1024x32x32xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x1024x32x32xf16>

}
