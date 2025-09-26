//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOHOverlapped
func.func @ConvAssignedSOHOverlapped(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoConvsAssignedSOHOverlapped
func.func @TwoConvsAssignedSOHOverlapped(%arg0: tensor<1x256x12x4xf16, {order = #NHWC}>) -> tensor<1x64x6x2xf16, {order = #NHWC}> {
    %weight_table_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weight_0= const.Declare tensor<32x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %weight_table_1 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %weight_1= const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weight_0, %weight_table_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [32, 256, 1, 1], strides = [2, 2]} : tensor<1x256x12x4xf16, {order = #NHWC}>, tensor<32x256x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x6x2xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weight_1, %weight_table_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]} : tensor<1x32x6x2xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x6x2xf16, {order = #NHWC}>
    return %1 : tensor<1x64x6x2xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE_0:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS_0:%.*]] = const.Declare tensor<32x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x256x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[WEIGHTSTABLE_1:%.*]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS_1:%.*]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[CONV_0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS_0]], [[WEIGHTSTABLE_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 256, 1, 1], strides = [2, 2]}
    //CHECK-SAME:        -> tensor<1x32x6x2xf16, {order = #NHWC}>
    // When calcuate the related dma cost for conv1, the compiler will back infer the conv0's input shape by its output,
    // for which the infered required output shape is  [1, 256, 11, 3] instead of orginal shape [1, 256, 12, 4], and the
    // related distributed type need use this shape to further generate the overlap parameter.
    //CHECK:        [[CONV_1:%.*]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS_1]], [[WEIGHTSTABLE_1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]}
    //CHECK-SAME:        -> tensor<1x64x6x2xf16, {order = #NHWC}>
    //CHECK:        return [[CONV_1]] : tensor<1x64x6x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOB3Batches
// CHECK-SAME:      [[INPUT:%.*]]: tensor<3x1024x14x14xf16, {order = #NHWC}>
func.func @ConvAssignedSOB3Batches(%arg0: tensor<3x1024x14x14xf16, {order = #NHWC}>) -> tensor<3x256x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} : tensor<3x1024x14x14xf16, {order = #NHWC}>, tensor<256x1024x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<3x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<3x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<3x256x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<3x256x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOB
// CHECK-SAME:      [[INPUT:%.*]]: tensor<6x1024x14x14xf16, {order = #NHWC}>
func.func @ConvAssignedSOB(%arg0: tensor<6x1024x14x14xf16, {order = #NHWC}>) -> tensor<6x256x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]} : tensor<6x1024x14x14xf16, {order = #NHWC}>, tensor<256x1024x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<6x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<6x256x14x14xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 1024, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<6x256x14x14xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<6x256x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedClustering
func.func @ConvAssignedClustering(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x48x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<48x64x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [48, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x48x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x48x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvAssignedSOHOverlapped
func.func @DepthConvAssignedSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedClustering
func.func @DepthConvAssignedClustering(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOHOverlapped
func.func @MaxPoolAssignedSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    //CHECK-SAME:   {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedClustering
func.func @MaxPoolAssignedClustering(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            kernel_size = [1, 1]
         } -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    //CHECK-SAME:   {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolAssignedSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOB3Batches(%input: tensor<3x32x112x112xf16, {order = #NHWC}>) -> tensor<3x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<3x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<3x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<3x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOB(%input: tensor<6x32x112x112xf16, {order = #NHWC}>) -> tensor<6x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<6x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<6x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<6x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AveragePoolAssignedSOB3Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x112x112xf16, {order = #NHWC}>)
func.func @AveragePoolAssignedSOB3Batches(%input: tensor<3x32x112x112xf16, {order = #NHWC}>) -> tensor<3x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<3x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<3x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<3x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[AVGPOOL]] : tensor<3x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AveragePoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<6x32x112x112xf16, {order = #NHWC}>)
func.func @AveragePoolAssignedSOB(%input: tensor<6x32x112x112xf16, {order = #NHWC}>) -> tensor<6x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<6x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<6x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[AVGPOOL]] : tensor<6x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOHOverlapped
func.func @SparseConvAssignedSOHOverlapped(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x64x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 64, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>>, tensor<80x1x1x4xsi32> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x64x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x640xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:           [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x64x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x640xi1>, is_weights, #VPU.SparsityCompression

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<1> : tensor<80x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [80, 64, 3, 3],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                                sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SparseConvAssignedSOK
func.func @SparseConvAssignedSOK(%arg0 : tensor<1x128x1x1xf16, {order = #NHWC}>, %arg1 : tensor<1x128x1x1xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00> : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>>

    %weights_table = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse, %weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>>, tensor<1008x1x1x4xsi32> -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<1008x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<1008x1x1x128xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1008x128x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:           [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression

    // CHECK-DAG:       [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]], [[CST_WEIGHTS_TABLE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [1008, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>> {
    // CHECK:         VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <VECTOR_FP16>
    // CHECK:       }

    // CHECK:       return [[OUT]] : !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                                     sparsity_map=tensor<1x1008x1x1xi1, {order = #NHWC}>>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedClustering
func.func @TanhAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %1 = VPU.Tanh(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultTANH]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClustering
func.func @MVNAssignedClustering(%arg0: tensor<1x1x1x512xf16, {order = #NCHW}>) -> tensor<1x1x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClusteringOn4
func.func @MVNAssignedClusteringOn4(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x4x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedSplitOverKernel
func.func @MVNAssignedSplitOverKernel(%arg0: tensor<1x12x1x512xf16, {order = #NCHW}>) -> tensor<1x12x1x512xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x12x1x512xf16, {order = #NCHW}> -> tensor<1x12x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x12x1x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x12x1x512xf16, {order = #NCHW}> -> tensor<1x12x1x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x12x1x512xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL: @Mvn6AssignedSplitOverKernel
func.func @Mvn6AssignedSplitOverKernel(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [2],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @Mvn6AssignedSplitOverHeight
func.func @Mvn6AssignedSplitOverHeight(%arg0: tensor<1x32x15x64xf16>) -> tensor<1x32x15x64xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 3], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    return %0 : tensor<1x32x15x64xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [1, 3],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x64xf16> -> tensor<1x32x15x64xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x64xf16>
}

// -----

// CHECK-LABEL: @Mvn6AssignedClustering
func.func @Mvn6AssignedClustering(%arg0: tensor<1x32x15x1xf16>) -> tensor<1x32x15x1xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [1, 2], eps = 9.9999997473787516E-6 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x32x15x1xf16> -> tensor<1x32x15x1xf16>
    return %0 : tensor<1x32x15x1xf16>

    //CHECK:   [[MVN6:%.+]] = VPU.MVN6({{[^:]+}}) {
    //CHECK-SAME:     axes = [1, 2],
    //CHECK-SAME:     eps = 9.9999997473787516E-6 : f64,
    //CHECK-SAME:     eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>,
    //CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    //CHECK-SAME:     normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    //CHECK-SAME:     : tensor<1x32x15x1xf16> -> tensor<1x32x15x1xf16>
    //CHECK:   return [[MVN6]] : tensor<1x32x15x1xf16>
}

// -----

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

!qElemType = !quant.uniform<u8<0:254>:f16:0, {0.001357270859357879:127,0.0016182628203564742:127,0.0020815957718946804:127,0.0026390524361077257:127,0.0010264177960673654:127,0.0028241668629834034:127,9.4622608244888429E-4:127,0.0012354281708950132:127,0.0010431431175216914:127,0.0020313619628665952:127,0.0023819006334139608:127,0.0036940675551497093:127,0.0040705110144427445:127,0.0010953865886673213:127,0.0024318298486274057:127,0.0024014103600359339:127,0.0039203847487141764:127,0.0018524924131828968:127,0.0026850329609367793:127,0.002006289292508223:127,0.0038565505207992915:127,0.0011652208219362994:127,0.0015601711714361597:127,0.0016786359895871381:127,0.0030253930824009454:127,0.0033942524373062012:127,0.0021159609002391186:127,0.0027468558840864285:127,0.003248042243672168:127,8.3886902398011813E-4:127,0.001087947741268188:127,0.0034467451215728999:127,0.0012410967134115264:127,0.0016006742875407062:127,8.0011086905096458E-4:127,0.0012080863000839715:127,6.9277856762953627E-4:127,0.0010478224341324933:127,0.0013355103534037672:127,0.0016338377017674484:127,0.0040408232080654839:127,0.0022144775221666952:127,0.0012906775699825738:127,0.0031165220136717547:127,0.0023295567260952447:127,0.0038008764972836955:127,0.0041627433356337656:127,0.0025403243819559652:127,0.0040322837867136077:127,7.5960124102164442E-4:127,0.0035432028019522117:127,0.0027394897825135959:127,0.0023073042471577804:127,0.0036451314377972462:127,0.0019629199908474297:127,0.0037665498538280097:127,7.3536362235001695E-4:127,0.0019541361669855794:127,0.0039462034157880651:127,0.0012378619881126824:127,9.717483689465861E-4:127,0.0013865374439344632:127,0.0026975860745888057:127,0.0011498732125665258:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.001357270859357879,0.0016182628203564742,0.0020815957718946804,0.0026390524361077257,0.0010264177960673654,0.0028241668629834034,9.4622608244888429E-4,0.0012354281708950132,0.0010431431175216914,0.0020313619628665952,0.0023819006334139608,0.0036940675551497093,0.0040705110144427445,0.0010953865886673213,0.0024318298486274057,0.0024014103600359339,0.0039203847487141764,0.0018524924131828968,0.0026850329609367793,0.002006289292508223,0.0038565505207992915,0.0011652208219362994,0.0015601711714361597,0.0016786359895871381,0.0030253930824009454,0.0033942524373062012,0.0021159609002391186,0.0027468558840864285,0.003248042243672168,8.3886902398011813E-4,0.001087947741268188,0.0034467451215728999,0.0012410967134115264,0.0016006742875407062,8.0011086905096458E-4,0.0012080863000839715,6.9277856762953627E-4,0.0010478224341324933,0.0013355103534037672,0.0016338377017674484,0.0040408232080654839,0.0022144775221666952,0.0012906775699825738,0.0031165220136717547,0.0023295567260952447,0.0038008764972836955,0.0041627433356337656,0.0025403243819559652,0.0040322837867136077,7.5960124102164442E-4,0.0035432028019522117,0.0027394897825135959,0.0023073042471577804,0.0036451314377972462,0.0019629199908474297,0.0037665498538280097,7.3536362235001695E-4,0.0019541361669855794,0.0039462034157880651,0.0012378619881126824,9.717483689465861E-4,0.0013865374439344632,0.0026975860745888057,0.0011498732125665258}>
!qElemType3 = !quant.uniform<u8:f16, 0.019849412581499887>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvBeforeMVNAssignedSplitOverKernel
func.func @ConvBeforeMVNAssignedSplitOverKernel(%arg0: tensor<1x128x64x128x!qElemType3, {order = #NHWC}>) -> tensor<1x64x64x128xf16> {
    %cst = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x128x4x4x!qElemType, {order = #NHWC}> = dense<10.1> : tensor<64x128x4x4xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %4 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 4, 4], strides = [1, 1]} : tensor<1x128x64x128x!qElemType3, {order = #NHWC}>, tensor<64x128x4x4x!qElemType, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x64x128xf16>
    %5 = VPU.MVN(%4) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %5 : tensor<1x64x64x128xf16>

    //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK:    [[MVN:%.+]] = VPU.MVN([[CONV]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:    return [[MVN]] : tensor<1x64x64x128xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanhAfterConvAssignedClustering
func.func @TanhAfterConvAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

    %39 = const.Declare tensor<1x1024x1x1xf16> = dense<10.1> : tensor<1x1024x1x1xf16>
    %40 = const.Declare tensor<1x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1x1xf16, {order = #NHWC}>

    %cst_7 = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1024xf16>, [#const.Reshape<[1024, 1024]>, #const.Reshape<[1024, 1024, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_27 = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>

    %42 = VPU.NCE.Convolution(%40, %cst_7, %cst_27) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} : tensor<1x1024x1x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %45 = VPU.Tanh(%42) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    return %45 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:   [[ResultConvolution:%.*]] = VPU.NCE.Convolution([[cst0:%.*]], [[cst1:%.*]], [[cst2:%.*]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   [[ResultTanh:%.*]] = VPU.Tanh([[ResultConvolution]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    //CHECK:   return [[ResultTanh]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOHOverlapped
func.func @CompressConvolutionAssignedSOHOverlapped(%arg0: tensor<1x3x224x224xf16, {order = #NHWC}>) -> tensor<1x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table)
            {
                cm_sp_pattern = 15 : i64,
                pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
            } -> tensor<1x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    //CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    //CHECK:        [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    //CHECK-SAME:           tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    //CHECK-SAME:           cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
    //CHECK-SAME:           rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
    //CHECK-SAME:           -> tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x64x112x112xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOB3Batches
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<3x3x224x224xf16, {order = #NHWC}>)
func.func @CompressConvolutionAssignedSOB3Batches(%arg0: tensor<3x3x224x224xf16, {order = #NHWC}>) -> tensor<3x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<3x3x224x224xf16, {order = #NHWC}> -> tensor<3x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table) {
        cm_sp_pattern = 7 : i64,
        pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 3, 7, 7], strides = [2, 2]
    } -> tensor<3x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<3x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    // CHECK-SAME:  tensor<3x3x224x224xf16, {order = #NHWC}> -> tensor<3x4x224x224xf16, {order = #NHWC}>

    // CHECK:   [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:   {cm_sp_pattern = 7 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,

    // CHECK:    return [[VAL0]] : tensor<3x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOB
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<6x3x224x224xf16, {order = #NHWC}>)
func.func @CompressConvolutionAssignedSOB(%arg0: tensor<6x3x224x224xf16, {order = #NHWC}>) -> tensor<6x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<6x3x224x224xf16, {order = #NHWC}> -> tensor<6x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table) {
        cm_sp_pattern = 7 : i64,
        pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 3, 7, 7], strides = [2, 2]
    } -> tensor<6x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<6x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    // CHECK-SAME:  tensor<6x3x224x224xf16, {order = #NHWC}> -> tensor<6x4x224x224xf16, {order = #NHWC}>

    // CHECK:   [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:   {cm_sp_pattern = 7 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,

    // CHECK:    return [[VAL0]] : tensor<6x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Track E#160175 - SEP cost support

// CHECK-LABEL: @InterpolateNearestAssignedSOH
func.func @InterpolateNearestAssignedSOH(%arg0: tensor<1x128x10x10xf16, {order = #NHWC}>) -> tensor<1x128x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x128x20x20xi1> = dense<1> : tensor<1x128x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [128], dataShape = [1, 128, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 20, 20]>
    } -> tensor<1x1x20x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 20, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x128x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x128x20x20xi1>,
                           storage_element_table=tensor<1x1x20x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 20, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [128, 128, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x128x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x128x20x20xi1> = dense<true> : tensor<1x128x20x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>
    // CHECK-SAME:      rawFilterShape = [128, 128, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x128x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedSOK
func.func @InterpolateNearestAssignedSOK(%arg0: tensor<1x256x10x10xf16, {order = #NHWC}>) -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x20x20xi1> = dense<1> : tensor<1x256x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [256], dataShape = [1, 256, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 256, 20, 20]>
    } -> tensor<1x1x20x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 256, 20, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x256x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x256x20x20xi1>,
                           storage_element_table=tensor<1x1x20x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 256, 20, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [256, 256, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x256x20x20xi1> = dense<true> : tensor<1x256x20x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>
    // CHECK-SAME:      rawFilterShape = [256, 256, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Track E#160175 - SEP cost support

// CHECK-LABEL: @InterpolateBilinearAssignedSOH
func.func @InterpolateBilinearAssignedSOH(%arg0: tensor<1x96x20x20xf16, {order = #NHWC}>) -> tensor<1x96x40x40xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x96x41x41xi1> = dense<1> : tensor<1x96x41x41xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 96, 41, 41]>
    } -> tensor<1x1x41x41xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 96, 41, 41]>
    } -> !VPU.SparseTensor<data=tensor<1x96x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x41x41xi1>,
                           storage_element_table=tensor<1x1x41x41xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 96, 41, 41]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [96, 96, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x96x40x40xf16, {order = #NHWC}>

    return %interpolate : tensor<1x96x40x40xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x96x41x41xi1> = dense<true> : tensor<1x96x41x41xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>
    // CHECK-SAME:      rawFilterShape = [96, 96, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x40x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOK
func.func @InterpolateBilinearAssignedSOK(%arg0: tensor<1x256x10x10xf16, {order = #NHWC}>) -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x21x21xi1> = dense<1> : tensor<1x256x21x21xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [256], dataShape = [1, 256, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 256, 21, 21]>
    } -> tensor<1x1x21x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 256, 21, 21]>
    } -> !VPU.SparseTensor<data=tensor<1x256x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x256x21x21xi1>,
                           storage_element_table=tensor<1x1x21x21xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 256, 21, 21]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [256, 256, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x256x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x256x21x21xi1> = dense<true> : tensor<1x256x21x21xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>
    // CHECK-SAME:      rawFilterShape = [256, 256, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignSOHForLayerWithLargeActivation
func.func @AssignSOHForLayerWithLargeActivation(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>) -> tensor<1x32x250x250xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} : tensor<1x64x250x250xf16, {order = #NHWC}>, tensor<32x64x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    return %0 : tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK-SAME:        rawFilterShape = [32, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:      -> tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x250x250xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverHeight
func.func @SoftMaxAssignedSplitOverHeight(%arg0: tensor<1x4x8x512xf16, {order = #NCHW}>) -> tensor<1x4x8x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x4x8x512xf16, {order = #NCHW}> -> tensor<1x4x8x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x8x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x8x512xf16, {order = #NCHW}> -> tensor<1x4x8x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x8x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverWidth
func.func @SoftMaxAssignedSplitOverWidth(%arg0: tensor<1x1x2x512xf16, {order = #NCHW}>) -> tensor<1x1x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 2} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x1x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x1x2x512xf16, {order = #NCHW}> -> tensor<1x1x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x1x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverKernel
func.func @SoftMaxAssignedSplitOverKernel(%arg0: tensor<1x6x2x512xf16, {order = #NCHW}>) -> tensor<1x6x2x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 2} : tensor<1x6x2x512xf16, {order = #NCHW}> -> tensor<1x6x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x6x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x6x2x512xf16, {order = #NCHW}> -> tensor<1x6x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x6x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedClustering
func.func @SoftMaxAssignedClustering(%arg0: tensor<1x4x1x1xf16, {order = #NCHW}>) -> tensor<1x4x1x1xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x4x1x1xf16, {order = #NCHW}> -> tensor<1x4x1x1xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x1xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x4x1x1xf16, {order = #NCHW}> -> tensor<1x4x1x1xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x1x1xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceAssignedSplitOverHeight
func.func @DepthToSpaceAssignedSplitOverHeight(%arg0: tensor<1x128x180x270xf16, {order = #NHWC}>) -> tensor<1x8x720x1080xf16, {order = #NHWC}> {

    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x720x1080xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x8x720x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceAssignedSplitOverWidth
func.func @DepthToSpaceAssignedSplitOverWidth(%arg0: tensor<1x128x1x270xf16, {order = #NHWC}>) -> tensor<1x8x4x1080xf16, {order = #NHWC}> {

    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>

    return %0 : tensor<1x8x4x1080xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>} : tensor<1x128x1x270xf16, {order = #NHWC}> -> tensor<1x8x4x1080xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x8x4x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ParentPermuteOpWithHeightEqualOneNoStrategy
func.func @ParentPermuteOpWithHeightEqualOneNoStrategy(%arg0: tensor<1x16x1x112xf16>, %arg1: tensor<1x16x32x112xf16>) -> tensor<1x16x33x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x1x112xf16, {order = #NHWC}>
    %1 = VPU.NCE.AveragePool(%0) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x112xf16, {order = #NHWC}>

    %2 = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    %3 = VPU.NCE.AveragePool(%2) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x32x112xf16, {order = #NHWC}>

    %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x16x1x112xf16, {order = #NHWC}>, tensor<1x16x32x112xf16, {order = #NHWC}> -> tensor<1x16x33x112xf16, {order = #NHWC}>

    return %4 : tensor<1x16x33x112xf16, {order = #NHWC}>

    //CHECK: [[Permute_1:%.*]] = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x1x112xf16, {order = #NHWC}>
    //CHECK: [[AVGPool_1:%.*]] = VPU.NCE.AveragePool([[Permute_1]]) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x112xf16, {order = #NHWC}>

    //CHECK: [[Permute_2:%.*]] = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEStub<>} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    //CHECK: [[AVGPool_2:%.*]] = VPU.NCE.AveragePool([[Permute_2]]) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    //CHECK: [[Result:%.*]] = VPU.Concat([[AVGPool_1]], [[AVGPool_2]]) {
    //CHECK-SAME{LITERAL}:            static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]
    //CHECK:  } : tensor<1x16x1x112xf16, {order = #NHWC}>, tensor<1x16x32x112xf16, {order = #NHWC}> -> tensor<1x16x33x112xf16, {order = #NHWC}>

    //CHECK: return [[Result]] : tensor<1x16x33x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ChildPermuteOpWithHeightEqualOneNoStrategy
func.func @ChildPermuteOpWithHeightEqualOneNoStrategy(%arg0: tensor<1x512x1x16xf16>, %arg1: tensor<1x512x1x32xf16>) -> tensor<1x512x1x48xf16, {order = #NHWC}> {
    %0 = VPU.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]} : tensor<1x512x1x16xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x48xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 512 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x512x1x48xf16, {order = #NHWC}>
    return %1 : tensor<1x512x1x48xf16, {order = #NHWC}>


    //CHECK: [[Concat:%.*]] = VPU.Concat(%arg0, %arg1) {
    //CHECK-SAME{LITERAL}:            static_offsets = [[0, 0, 0, 0], [0, 0, 0, 16]]
    //CHECK:  } : tensor<1x512x1x16xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x48xf16>

    //CHECK: [[Permute:%.*]] = VPU.NCE.Permute([[Concat]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 512 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x512x1x48xf16, {order = #NHWC}>
    //CHECK: return [[Permute]] : tensor<1x512x1x48xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @PermuteAssignedSOHOverlapped
func.func @PermuteAssignedSOHOverlapped(%arg0: tensor<1x3x256x224xf16>) -> tensor<1x4x256x224x!qElemType, {order = #NHWC}> {
    %PERMUTE = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x256x224x!qElemType, {order = #NHWC}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>

    return %PERMUTE : tensor<1x4x256x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PermuteFP16AssignedSOHOverlapped
func.func @PermuteFP16AssignedSOHOverlapped(%arg0: tensor<1x3x256x224xf16>) -> tensor<1x4x256x224xf16, {order = #NHWC}> {
    %PERMUTE = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x256x224xf16, {order = #NHWC}>

    // CHECK:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>

    return %PERMUTE : tensor<1x4x256x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ClampAssignedSplitOverKernel
func.func @ClampAssignedSplitOverKernel(%arg0: tensor<1x16x1x60xf16, {order = #NHWC}>) -> tensor<1x16x1x60xf16, {order = #NHWC}> {

    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} :
        tensor<1x16x1x60xf16, {order = #NHWC}> -> tensor<1x16x1x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x60xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.Clamp(%arg0) {
    //CHECK-SAME:       max = 1.000000e+00 : f64, min = -1.000000e+00 : f64,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
    //CHECK-SAME:       tensor<1x16x1x60xf16, {order = #NHWC}> -> tensor<1x16x1x60xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x16x1x60xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ClampAssignedSplitOverHeight
func.func @ClampAssignedSplitOverHeight(%arg0: tensor<1x16x128x60xf16, {order = #NHWC}>) -> tensor<1x16x128x60xf16, {order = #NHWC}> {

    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} :
        tensor<1x16x128x60xf16, {order = #NHWC}> -> tensor<1x16x128x60xf16, {order = #NHWC}>

    return %0 : tensor<1x16x128x60xf16, {order = #NHWC}>

    //CHECK:   [[Result:%.*]] = VPU.Clamp(%arg0) {
    //CHECK-SAME:       max = 1.000000e+00 : f64, min = -1.000000e+00 : f64,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
    //CHECK-SAME:       tensor<1x16x128x60xf16, {order = #NHWC}> -> tensor<1x16x128x60xf16, {order = #NHWC}>
    //CHECK:   return [[Result]] : tensor<1x16x128x60xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @DontAssignStrategy2DConvert
func.func @DontAssignStrategy2DConvert(%arg0: tensor<10x2xf32>) -> tensor<10x2xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<10x2xf32> -> tensor<10x2xf16>
    return %0 : tensor<10x2xf16>
    //CHECK:   [[Result:%.*]] = VPU.Convert([[ARG:%.*]]) {dstElemType = f16} : tensor<10x2xf32> -> tensor<10x2xf16>
    //CHECK:   return [[Result]] : tensor<10x2xf16>
}

// -----

// CHECK-LABEL: @AbsAssignedSplitOverKernel
func.func @AbsAssignedSplitOverKernel(%arg0: tensor<1x16x1x60xf16>) -> tensor<1x16x1x60xf16> {

    %0 = VPU.Abs(%arg0) : tensor<1x16x1x60xf16> -> tensor<1x16x1x60xf16>

    return %0 : tensor<1x16x1x60xf16>

    //CHECK:   [[ABS:%.+]] = VPU.Abs(%arg0) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x16x1x60xf16> -> tensor<1x16x1x60xf16>
    //CHECK:   return [[ABS]] : tensor<1x16x1x60xf16>
}

// -----

// CHECK-LABEL: @AbsAssignedSplitOverHeight
func.func @AbsAssignedSplitOverHeight(%arg0: tensor<1x1x128x60xf16>) -> tensor<1x1x128x60xf16> {

    %0 = VPU.Abs(%arg0) : tensor<1x1x128x60xf16> -> tensor<1x1x128x60xf16>

    return %0 : tensor<1x1x128x60xf16>

    //CHECK:   [[ABS:%.+]] = VPU.Abs(%arg0) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x128x60xf16> -> tensor<1x1x128x60xf16>
    //CHECK:   return [[ABS]] : tensor<1x1x128x60xf16>
}

// -----

// CHECK-LABEL: @DivideAssignedSplitOverKernel
func.func @DivideAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[DIVIDE:%.+]] = VPU.Divide(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[DIVIDE]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL: @DivideAssignedSplitOverHeight
func.func @DivideAssignedSplitOverHeight(%arg0: tensor<1x16x256x256xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x256x256xf16> {

    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>

    return %0 : tensor<1x16x256x256xf16>

    //CHECK:   [[DIVIDE:%.+]] = VPU.Divide(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
    //CHECK:   return [[DIVIDE]] : tensor<1x16x256x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedSplitOverKernel
func.func @PowerAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[POWER]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedSplitOverHeight
func.func @PowerAssignedSplitOverHeight(%arg0: tensor<1x16x256x256xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x256x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>

    return %0 : tensor<1x16x256x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x16x256x256xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x256x256xf16>
    //CHECK:   return [[POWER]] : tensor<1x16x256x256xf16>
}

// -----

// CHECK-LABEL: @PowerAssignedClustering
func.func @PowerAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[POWER:%.+]] = VPU.Power(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[POWER]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @GreaterAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @GreaterAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[GREATER]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @GreaterAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @GreaterAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[GREATER]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @GreaterAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @GreaterAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[GREATER:%.+]] = VPU.Greater(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[GREATER]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @LessAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[LESS]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @LessAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[LESS]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @LessAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @LessAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[LESS:%.+]] = VPU.Less(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[LESS]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @EqualAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @EqualAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedSplitOverWidth
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x1152xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @EqualAssignedSplitOverWidth(%arg0: tensor<1x1x1x1152xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1152xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1152xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1152xi8>

    return %0 : tensor<1x1x1x1152xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>}
    //CHECK-SAME:       tensor<1x1x1x1152xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1152xi8>
    //CHECK:   return [[EQUAL]] : tensor<1x1x1x1152xi8>
}

// -----

// CHECK-LABEL:   @EqualAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @EqualAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[EQUAL:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[EQUAL]] : tensor<2x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @NotEqualAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @NotEqualAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xi8> {

    %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>

    return %0 : tensor<1x106x1x256xi8>

    //CHECK:   [[NOTEQUAL:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xi8>
    //CHECK:   return [[NOTEQUAL]] : tensor<1x106x1x256xi8>
}

// -----

// CHECK-LABEL:   @NotEqualAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @NotEqualAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xi8> {

    %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>

    return %0 : tensor<1x1x256x256xi8>

    //CHECK:   [[NOTEQUAL:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xi8>
    //CHECK:   return [[NOTEQUAL]] : tensor<1x1x256x256xi8>
}

// -----

// CHECK-LABEL:   @NotEqualAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @NotEqualAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xi8> {

    %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>

    return %0 : tensor<2x106x1x256xi8>

    //CHECK:   [[NOTEQUAL:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xi8>
    //CHECK:   return [[NOTEQUAL]] : tensor<2x106x1x256xi8>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipSubGraphOptimizationForINFCost
func.func @SkipSubGraphOptimizationForINFCost(%input_data: tensor<1x128x72x72xf16, {order = #NHWC}>, %input_data2: tensor<1x64x144x144xf16, {order = #NHWC}>) -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    // conv
    %convWeights = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %convWeightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input_data, %convWeights, %convWeightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x72x72xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // transposedConv
    %sparsityMap = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        dataShape = [1, 64, 72, 72],
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>,
        seDepth = 1 : i64,
        seSize = [64]
    } -> tensor<1x1x147x147xi32, {order = #NHWC}>
    %input = VPU.GroupSparseTensor(%conv, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEUpsampling<
            factors = [1, 1],
            padding = [2, 2, 2, 2]>
    } -> !VPU.SparseTensor<
            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
            #VPU.SEUpsampling<
                factors = [1, 1],
                padding = [2, 2, 2, 2]>>

    %weightsCst = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weightsSparsityMap = const.Declare tensor<64x1x1x1024xi1> = dense<1.0> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %weights = VPU.GroupSparseTensor(%weightsCst, %weightsSparsityMap) {
        sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>

    %weightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %transposedConv = VPU.NCE.Convolution(%input, %weights, %weightsTable) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]
    } : !VPU.SparseTensor< data=tensor<1x64x72x72xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>, #VPU.SEUpsampling< factors = [1, 1], padding = [2, 2, 2, 2]>>, !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>, tensor<64x1x1x4xsi32> -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // add
    %add = VPU.NCE.Eltwise(%transposedConv, %input_data2) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>


    return %add : tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[CONV_WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:        [[CONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[CONV_WEIGHTS]], [[CONV_WEIGHTS_TBL]])
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]}
    // CHECK-SAME:        -> tensor<1x64x72x72xf16, {order = #NHWC}>

    // CHECK:        [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x147x147xi1, {order = #NHWC}> = dense<1> : tensor<1x64x147x147xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    // CHECK:        [[SE_TBL:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:        {dataElemType = f16,
    // CHECK-SAME:        dataShape = [1, 64, 72, 72],
    // CHECK-SAME:        seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>,
    // CHECK-SAME:            seDepth = 1 : i64, seSize = [64]} -> tensor<1x1x147x147xi32, {order = #NHWC}>
    // CHECK:        [[SPARSE_INPUT:%.+]] = VPU.GroupSparseTensor([[CONV]], [[SPARSITY_MAP]], [[SE_TBL]])
    // CHECK-SAME:        {seAttr = #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>}
    // CHECK-SAME:        -> !VPU.SparseTensor<
    // CHECK-SAME:            data=tensor<1x64x72x72xf16, {order = #NHWC}>,
    // CHECK-SAME:            sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>,
    // CHECK-SAME:            storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>,
    // CHECK-SAME:        #VPU.SEUpsampling<
    // CHECK-SAME:            factors = [1, 1],
    // CHECK-SAME:            padding = [2, 2, 2, 2]>>
    // CHECK:        [[DECONV_WEIGHTS_CST:%.+]] = const.Declare tensor<64x64x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    // CHECK:        [[DECONV_WEIGHTS_SM:%.+]] = const.Declare tensor<64x1x1x1024xi1> = dense<1.000000e+00> : tensor<64x64x4x4xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    // CHECK:        [[DECONV_WEIGHTS:%.+]] = VPU.GroupSparseTensor([[DECONV_WEIGHTS_CST]], [[DECONV_WEIGHTS_SM]]) {
    // CHECK-SAME:        is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>}
    // CHECK-SAME:        -> !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>>
    // CHECK:        [[DECONV_WEIGHTS_TBL:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[DECONV:%.+]] = VPU.NCE.Convolution([[SPARSE_INPUT]], [[DECONV_WEIGHTS]], [[DECONV_WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]}
    // CHECK-SAME:        -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[ADD:%.+]] = VPU.NCE.Eltwise([[DECONV]], %arg1)
    // CHECK-SAME:        {is_inplace = true,
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>

    // CHECK-SAME:        -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        return [[ADD]] : tensor<1x64x144x144xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   @SubtractAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @SubtractAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @SubtractAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @SubtractAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @SubtractAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @SubtractAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[SUBTRACT:%.+]] = VPU.Subtract(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[SUBTRACT]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @AddAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[ADD]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @AddAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[ADD]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @AddAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @AddAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[ADD:%.+]] = VPU.Add(%arg0, %arg1) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[ADD]] : tensor<2x106x1x256xf16>

}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SingleTileConvert
// CHECK-SAME:    [[INPUT_DATA0:%.+]]: tensor<1x4x3x3xf32, {order = #NHWC}>
func.func @SingleTileConvert(%arg0: tensor<1x4x3x3xf32, {order = #NHWC}>) -> tensor<1x1x6x6xf16, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x4x3x3xf32, {order = #NHWC}> -> tensor<1x4x3x3xf16, {order = #NHWC}>
    %1 = VPU.DepthToSpace(%0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x4x3x3xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>
    return %1: tensor<1x1x6x6xf16, {order = #NHWC}>

    // VPU.Convert is firstly assigned as clustering strategy and then the strategy is removed
    // since the following VPU.DepthToSpace op doesn't have a strategy
    // CHECK:          [[CONVERT:%.*]] = VPU.Convert([[INPUT_DATA0]]) {dstElemType = f16} : tensor<1x4x3x3xf32, {order = #NHWC}> -> tensor<1x4x3x3xf16, {order = #NHWC}>
    // CHECK:          [[DepthToSpace:%.*]] = VPU.DepthToSpace([[CONVERT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x4x3x3xf16, {order = #NHWC}> -> tensor<1x1x6x6xf16, {order = #NHWC}>
    // CHECK:          return [[DepthToSpace]] : tensor<1x1x6x6xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedSplitOverKernel
func.func @FloorAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedSplitOverHeight
func.func @FloorAssignedSplitOverHeight(%arg0: tensor<1x16x16x512xf16, {order = #NCHW}>) -> tensor<1x16x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x16x16x512xf16, {order = #NCHW}> -> tensor<1x16x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x16x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x16x512xf16, {order = #NCHW}> -> tensor<1x16x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @FloorAssignedClustering
func.func @FloorAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Floor(%arg0) : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Floor({{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedSplitOverKernel
func.func @RoundAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {
    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedSplitOverHeight
func.func @RoundAssignedSplitOverHeight(%arg0: tensor<1x1x16x512xf16, {order = #NCHW}>) -> tensor<1x1x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @RoundAssignedClustering
func.func @RoundAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Round({{[^:]+}}) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverKernel
func.func @LogAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverHeight
func.func @LogAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf16, {order = #NCHW}>) -> tensor<1x1x16x513xf16, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverKernel
func.func @SqrtAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverHeight
func.func @SqrtAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf16, {order = #NCHW}>) -> tensor<1x1x16x513xf16, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf16, {order = #NCHW}> -> tensor<1x1x16x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverKernel
func.func @LogAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf32, {order = #NCHW}>) -> tensor<1x16x1x513xf32, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogAssignedSplitOverHeight
func.func @LogAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf32, {order = #NCHW}>) -> tensor<1x1x16x513xf32, {order = #NCHW}> {

    %0 = VPU.Log(%arg0) : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Log(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverKernel
func.func @SqrtAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf32, {order = #NCHW}>) -> tensor<1x16x1x513xf32, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf32, {order = #NCHW}> -> tensor<1x16x1x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SqrtAssignedSplitOverHeight
func.func @SqrtAssignedSplitOverHeight(%arg0: tensor<1x1x16x513xf32, {order = #NCHW}>) -> tensor<1x1x16x513xf32, {order = #NCHW}> {

    %0 = VPU.Sqrt(%arg0) : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>

    return %0 : tensor<1x1x16x513xf32, {order = #NCHW}>

    //CHECK:   [[Result:%.*]] = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x513xf32, {order = #NCHW}> -> tensor<1x1x16x513xf32, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x513xf32, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceL1SplitOverKernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL1SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceL1(%[[VAL_0]]) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceL2SplitOverKernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
func.func @ReduceL2SplitOverKernel(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1024x1x1xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
  return %0 : tensor<1x1024x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceL2(%[[VAL_0]]) {axes_value = [2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1024x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1024x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceLogicalAndClustering(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalAndClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceLogicalAnd(%[[VAL_0]]) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceLogicalOrClustering(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
func.func @ReduceLogicalOrClustering(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x1x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 2, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
  return %0 : tensor<1x1x1x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceLogicalOr(%[[VAL_0]]) {axes_value = [1, 2, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x1x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceMaxSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMaxSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceMax(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceMeanSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceMeanSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceMean(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceProdSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceProdSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceProd(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceSumSplitOverHeight(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
func.func @ReduceSumSplitOverHeight(%arg0: tensor<1x1024x7x7xf16>) -> tensor<1x1x7x1xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
  return %0 : tensor<1x1x7x1xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceSum(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
// CHECK-SAME:    : tensor<1x1024x7x7xf16> -> tensor<1x1x7x1xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   func.func @ReduceSum5D(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<1x1024x7x7x3xf16>) -> tensor<1x1x7x1x3xf16> {
func.func @ReduceSum5D(%arg0: tensor<1x1024x7x7x3xf16>) -> tensor<1x1x7x1x3xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3], keep_dims} : tensor<1x1024x7x7x3xf16> -> tensor<1x1x7x1x3xf16>
  return %0 : tensor<1x1x7x1x3xf16>

// CHECK:       %[[VAL_1:.*]] = VPU.ReduceSum(%[[VAL_0]]) {axes_value = [1, 3], keep_dims, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
// CHECK-SAME:    : tensor<1x1024x7x7x3xf16> -> tensor<1x1x7x1x3xf16>
// CHECK:       return %[[VAL_1]] : tensor<1x1x7x1x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.01:124>
!qElemType1 = !quant.uniform<u8:f16, 0.02:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateDistributedQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>)
func.func @PropagateDistributedQuantizeCast(%arg0: tensor<1x512x14x14x!qElemType, {order = #NHWC}>) -> tensor<1x512x14x14x!qElemType1, {order = #NHWC}>{
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg0) { op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} :
         tensor<1x512x14x14x!qElemType, {order = #NHWC}>, tensor<1x512x14x14x!qElemType, {order = #NHWC}>
            -> tensor<1x512x14x14x!qElemType, {order = #NHWC}>

    %quant_cast = VPU.QuantizeCast(%eltwise) {dstElemType = !qElemType1} : tensor<1x512x14x14x!qElemType, {order = #NHWC}> -> tensor<1x512x14x14x!qElemType1, {order = #NHWC}>
    %weights2 = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
    %weightsTable2 = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%quant_cast, %weights2, %weightsTable2) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [512, 512, 1, 1], strides = [1, 1]}
            : tensor<1x512x14x14x!qElemType1, {order = #NHWC}>, tensor<512x512x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x14x14x!qElemType1, {order = #NHWC}>

    return %conv : tensor<1x512x14x14x!qElemType1, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
    // CHECK:       [[CAST:%.+]] = VPU.QuantizeCast
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<512x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x512x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:   [[WT:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[CAST]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: func.func @NoStrategyDetectionOutputSortForHeightLessThanTileNumber
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x2x10112xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1x2x10112xsi32>,
// CHECK-SAME:        [[INPUT3:%arg[0-9]]]: tensor<1x1x32x256xsi32>
func.func @NoStrategyDetectionOutputSortForHeightLessThanTileNumber(%arg0: tensor<1x1x2x10112xf16>, %arg1: tensor<1x1x2x10112xsi32>, %arg2: tensor<1x1x32x256xsi32>) -> (tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x2x1xsi32>){
    %confidence, %indices, %sizes = VPU.DetectionOutputSort(%arg0, %arg1, %arg2) {confidence_threshold = 0.0099999997764825821 : f64, top_k = 400 : i64} : tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x32x256xsi32> -> tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x2x1xsi32>

    return %confidence, %indices, %sizes : tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x2x1xsi32>

    // CHECK:        [[CONFIDENCE:%.+]], [[INDICES:%.+]], [[SIZE:%.+]] = VPU.DetectionOutputSort([[INPUT1]], [[INPUT2]], [[INPUT3]])
    // CHECK-SAME:               {confidence_threshold = 0.0099999997764825821 : f64, top_k = 400 : i64} : tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x32x256xsi32> -> tensor<1x1x2x10112xf16>, tensor<1x1x2x10112xsi32>, tensor<1x1x2x1xsi32>
    // CHECK:        return [[CONFIDENCE]], [[INDICES]], [[SIZE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotInferDistributedTypeOfPermuteCast
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x16x320x256xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x16x320x256xf16, {order = #NHWC}>
func.func @NotInferDistributedTypeOfPermuteCast(%arg0: tensor<1x16x320x256xf16, {order = #NHWC}>, %arg1: tensor<1x16x320x256xf16, {order = #NHWC}>) -> tensor<1x4096x320x1xf16>{
    %eltwise = VPU.NCE.Eltwise(%arg0, %arg1) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x320x256xf16, {order = #NHWC}>

    %shape_cast = VPU.ShapeCast {shape = [1, 1, 4096, 320]} inputs(%eltwise : tensor<1x16x320x256xf16, {order = #NHWC}>) -> tensor<1x1x4096x320xf16, {order = #NHWC}>

    %permute_cast = VPU.PermuteCast(%shape_cast) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x1x4096x320xf16, {order = #NHWC}> -> tensor<1x1x4096x320xf16>

    %reshape = VPU.AffineReshape(%permute_cast) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x1x4096x320xf16> -> tensor<1x4096x320x1xf16>

    %mvn = VPU.MVN(%reshape) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x4096x320x1xf16> -> tensor<1x4096x320x1xf16>

    return %mvn : tensor<1x4096x320x1xf16>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[SHAPE_CAST:%.+]] = VPU.ShapeCast
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPU.PermuteCast([[SHAPE_CAST]])
    // CHECK:       [[MVN:%.+]] = VPU.MVN
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    // CHECK:       return [[MVN]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0022039675245098039:131>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CWNH = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @PropagateDistributedPermuteCast
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x4096x1024x4xf16, {order = #NHWC}>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<48x4096x1x1xf16, {order = #NHWC}>
func.func @PropagateDistributedPermuteCast(%arg0: tensor<1x4096x1024x4xf16, {order = #NHWC}>, %arg1: tensor<48x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x320x1024x4x!qElemType, {order = #NHWC}> {
    %conv_weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %dw_conv_weights = const.Declare tensor<320x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x320xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 320, 1, 1]>, #const.Reshape<[320, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %dw_conv_weights_table = const.Declare tensor<320x1x1x4xsi32> = dense<1> : tensor<320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %arg1, %conv_weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]
    } : tensor<1x4096x1024x4xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x1024x4xf16, {order = #NHWC}>

    %1 = VPU.Slice %0 [0, 0, 0, 0] [1, 40, 1024, 4] : tensor<1x48x1024x4xf16, {order = #NHWC}> to tensor<1x40x1024x4xf16, {order = #NHWC}>

    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 40, 4096, 1]} : tensor<1x40x1024x4xf16, {order = #NHWC}> -> tensor<1x40x4096x1xf16, {order = #NHWC}>
    %3 = VPU.PermuteCast(%2) {dst_order = #NCHW, mem_perm = #CWNH} : tensor<1x40x4096x1xf16, {order = #NHWC}> -> tensor<4096x40x1x1xf16>
    %4 = VPU.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4096, 1, 40]} : tensor<4096x40x1x1xf16> -> tensor<1x4096x1x40xf16>
    %5 = VPU.PermuteCast(%4) {dst_order = #NHCW, mem_perm = #NCHW} : tensor<1x4096x1x40xf16> -> tensor<1x1x4096x40xf16, {order = #NHCW}>
    %6 = VPU.Concat(%5, %5, %5, %5, %5, %5, %5, %5) {
        static_offsets = [
            [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0],
            [0, 4, 0, 0], [0, 5, 0, 0], [0, 6, 0, 0], [0, 7, 0, 0]
        ]
    } : tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}>,
        tensor<1x1x4096x40xf16, {order = #NHCW}> -> tensor<1x8x4096x40xf16, {order = #NHCW}>

    %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x8x4096x40xf16, {order = #NHCW}> -> tensor<1x4096x8x40xf16>
    %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 320, 1]} : tensor<1x4096x8x40xf16> -> tensor<1x4096x320x1xf16>
    %9 = VPU.PermuteCast(%8) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x4096x320x1xf16> -> tensor<1x320x4096x1xf16, {order = #NHWC}>
    %10 = VPU.AffineReshape(%9) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 320, 1024, 4]} : tensor<1x320x4096x1xf16, {order = #NHWC}> -> tensor<1x320x1024x4xf16, {order = #NHWC}>
    %11 = VPU.NCE.DepthConvolution(%10, %dw_conv_weights, %dw_conv_weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [320, 1, 1, 1], strides = [1, 1]
    } -> tensor<1x320x1024x4x!qElemType, {order = #NHWC}>

    return %11 : tensor<1x320x1024x4x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[SLICE:%.+]] = VPU.Slice
    // CHECK:       [[RESHAPE1:%.+]] = VPU.AffineReshape
    // CHECK:       [[PERMUTECAST1:%.+]] = VPU.PermuteCast
    // CHECK:       [[RESHAPE2:%.+]] = VPU.AffineReshape
    // CHECK:       [[PERMUTECAST2:%.+]] = VPU.PermuteCast
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat
    // CHECK:       [[PERMUTECAST3:%.+]] = VPU.PermuteCast
    // CHECK:       [[RESHAPE3:%.+]] = VPU.AffineReshape
    // CHECK:       [[PERMUTECAST4:%.+]] = VPU.PermuteCast
    // CHECK:       [[RESHAPE4:%.+]] = VPU.AffineReshape
    // CHECK:       [[DWCONV:%.+]] = VPU.NCE.DepthConvolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK:       return [[DWCONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1NormalizeAssignedSplitOverHeight
func.func @MVN1NormalizeAssignedSplitOverHeight(%arg0: tensor<1x3x20x35971xf16>, %arg1: tensor<1x3x1x1xf16, {order = #NHWC}>) -> tensor<1x3x20x35971xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = false} : tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>
    return %0: tensor<1x3x20x35971xf16>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1Normalize(%arg0, %arg1)
    // CHECK-SAME:      {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = false} :
    // CHECK-SAME:      tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>

    //CHECK:        return [[VAL0]] : tensor<1x3x20x35971xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1MeanVarAssignedClustering
func.func @MVN1MeanVarAssignedClustering(%arg0: tensor<1x1x1x2xf32, {order = #NHWC}>) -> tensor<1x1x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
            across_channels = true,
            eps = 1.000000e-09 : f64,
            normalize_variance = true,
            orig_shape = [1, 1, 1, 515971],
            output_type = f16} :
        tensor<1x1x1x2xf32, {order = #NHWC}> -> tensor<1x1x1x2xf16, {order = #NHWC}>
    return %0: tensor<1x1x1x2xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1MeanVar(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>

    //CHECK:        return [[VAL0]] : tensor<1x1x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MVN1SumAssignedSplitOverKernel
func.func @MVN1SumAssignedSplitOverKernel(%arg0: tensor<1x12x20x3997xf16>) -> tensor<1x12x1x2xf32, {order = #NHWC}> {
    %0 = VPU.MVN1SumOp(%arg0) {
            across_channels = false,
            normalize_variance = true,
            output_height = 1 : i64} :
        tensor<1x12x20x3997xf16> -> tensor<1x12x1x2xf32, {order = #NHWC}>
    return %0: tensor<1x12x1x2xf32, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1SumOp(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    //CHECK:        return [[VAL0]] : tensor<1x12x1x2xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}
// CHECK-LABEL: @MVN1SumAssignedSOH
func.func @MVN1SumAssignedSOH(%arg0: tensor<1x3x3997x1xf16, {order = #NHWC}>) -> tensor<1x3x4x2xf32, {order = #NHWC}> {
    %0 = VPU.MVN1SumOp(%arg0) {
            across_channels = false,
            normalize_variance = true,
            output_height = 4 : i64} :
        tensor<1x3x3997x1xf16, {order = #NHWC}> -> tensor<1x3x4x2xf32, {order = #NHWC}>
    return %0: tensor<1x3x4x2xf32, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.MVN1SumOp(%arg0)
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        return [[VAL0]] : tensor<1x3x4x2xf32, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignSOCForPollingWithODUPermuteToNCXX
func.func @AssignSOCForPollingWithODUPermuteToNCXX(%arg0: tensor<1x3136x4x32xf16, {order = #NHWC}>) -> tensor<1x128x784x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    %cst_0 = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16, {order = #NHWC}>

    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x3136x4x32xf16>
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 784, 4, 128]} : tensor<1x3136x4x32xf16> -> tensor<1x784x4x128xf16>
    %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x784x4x128xf16> -> tensor<1x128x784x4xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x784x4xf16, {order = #NHWC}>, tensor<128x128x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x784x4xf16, {order = #NHWC}>

    return %3 : tensor<1x128x784x4xf16, {order = #NHWC}>

    //CHECK:        [[POOL:%.+]] = VPU.NCE.MaxPool
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:        [[RESHAPE:%.+]] = VPU.AffineReshape
    //CHECK:        [[PERMUTE:%.+]] = VPU.PermuteCast
    //CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:        return [[CONV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxAssignedSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x8x512xf16, {order = #NCHW}>)
func.func @LogSoftmaxAssignedSplitOverHeight(%arg0: tensor<1x4x8x512xf16, {order = #NCHW}>) -> tensor<1x4x8x512xf16, {order = #NCHW}> {
    %1 = VPU.LogSoftmax(%arg0) {axisInd = 3} : tensor<1x4x8x512xf16, {order = #NCHW}> -> tensor<1x4x8x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x8x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x8x512xf16, {order = #NCHW}> -> tensor<1x4x8x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x8x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxAssignedSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x36x2x512xf16, {order = #NCHW}>)
func.func @LogSoftmaxAssignedSplitOverKernel(%arg0: tensor<1x36x2x512xf16, {order = #NCHW}>) -> tensor<1x36x2x512xf16, {order = #NCHW}> {
    %1 = VPU.LogSoftmax(%arg0) {axisInd = 2} : tensor<1x36x2x512xf16, {order = #NCHW}> -> tensor<1x36x2x512xf16, {order = #NCHW}>

    return %1 : tensor<1x36x2x512xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x36x2x512xf16, {order = #NCHW}> -> tensor<1x36x2x512xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x36x2x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @LogSoftmaxAssignedClustering
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x1x1xf16, {order = #NCHW}>)
func.func @LogSoftmaxAssignedClustering(%arg0: tensor<1x4x1x1xf16, {order = #NCHW}>) -> tensor<1x4x1x1xf16, {order = #NCHW}> {
    %1 = VPU.LogSoftmax(%arg0) {axisInd = 1} : tensor<1x4x1x1xf16, {order = #NCHW}> -> tensor<1x4x1x1xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x1xf16, {order = #NCHW}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.LogSoftmax([[INPUT]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x4x1x1xf16, {order = #NCHW}> -> tensor<1x4x1x1xf16, {order = #NCHW}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x1x1xf16, {order = #NCHW}>

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @GroupedMatMulAssignedSOG
func.func @GroupedMatMulAssignedSOG(%arg0:  tensor<4x1x32x64x1xf16, {order =#GNHWC}>, %arg1: tensor<4x64x32x1x1xf16, {order = #GNHWC}>) -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    %cst = const.Declare tensor<4x64x1x1x4xsi32> = dense<1> : tensor<4x64x1x1x4xsi32>

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %cst) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [4, 1, 64, 32, 1], strides = [1, 1]
    } -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:        [[CONST:%.+]] = const.Declare
    // CHECK:        [[MATMUL:%.+]] = VPU.NCE.MatMul
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK:        return [[MATMUL]] : tensor<4x1x64x64x1xf16, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @ConvAssignedSOHWithSOHIncompatibleConsumer
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x256x2000x1xf16, {order = #NHWC}>)
func.func @ConvAssignedSOHWithSOHIncompatibleConsumer(%arg0: tensor<1x256x2000x1xf16, {order = #NHWC}>) -> tensor<1x256x2000x1xf16, {order = #NHWC}> {

    %weights_table_0 = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %weights_0 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights_0, %weights_table_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} : tensor<1x256x2000x1xf16, {order = #NHWC}>, tensor<256x256x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x1xf16, {order = #NHWC}>

    %weights_table_1 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %weights_1 = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %1 = VPU.NCE.Convolution(%0, %weights_1, %weights_table_1) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 256, 1, 1], strides = [8, 1]} : tensor<1x256x2000x1xf16, {order = #NHWC}>, tensor<64x256x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x250x1xf16, {order = #NHWC}>

    %2 = VPU.ReduceMean(%0) {axes_value = [2]} : tensor<1x256x2000x1xf16, {order = #NHWC}> -> tensor<1x256x1xf16, {order = #CHW}>

    return %0 : tensor<1x256x2000x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE0:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS0]], [[WEIGHTSTABLE0]])
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[WEIGHTSTABLE1:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]], [[WEIGHTSTABLE1]])
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        [[VAL1:%.+]] = VPU.ReduceMean([[VAL0]])
    //CHECK-NOT:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    //CHECK:        return [[VAL0]] : tensor<1x256x2000x1xf16, {order = #NHWC}>

}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @LSTMSequenceBidirectionalSplitOverKernel(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x80x512xf16>) -> (tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @LSTMSequenceBidirectionalSplitOverKernel(%arg0: tensor<1x2x80x512xf16>) -> (tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 80 : i64} : tensor<1x2x80x512xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x2x80x128xf16>, tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, sequenceLength = 80 : i64}
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @LSTMSequenceBidirectionalSplitOverBatch(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<3x2x80x512xf16>) -> (tensor<3x2x80x128xf16>, tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>) {
func.func @LSTMSequenceBidirectionalSplitOverBatch(%arg0: tensor<3x2x80x512xf16>) -> (tensor<3x2x80x128xf16>, tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<3x2x1x128xf16> = dense<1.000000e+00> : tensor<3x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, sequenceLength = 80 : i64} : tensor<3x2x80x512xf16>, tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>, tensor<2x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<3x2x80x128xf16>, tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<3x2x80x128xf16>, tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>, sequenceLength = 80 : i64}
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK-LABEL: func.func @LSTMSequenceForwardNoStrategy(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1x80x512xf16>) -> (tensor<1x1x80x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
func.func @LSTMSequenceForwardNoStrategy(%arg0: tensor<1x1x80x512xf16>) -> (tensor<1x1x80x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x1x1x128xf16> = dense<1.000000e+00> : tensor<1x1x1x128xf16>
    %cst_2 = const.Declare tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}> = dense<3.000000e+00> : tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>
    %cst_5 = const.Declare tensor<1x2x4x128xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x2x4x128xf16, {order = #NCWH}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst_5, %cst) {direction = #IE.rnn_seq_direction<FORWARD>, sequenceLength = 80 : i64} : tensor<1x1x80x512xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>, tensor<1x4x128x128xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>}>, tensor<1x2x4x128xf16, {order = #NCWH}>, tensor<1x1x1x2xsi32> -> tensor<1x1x80x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState : tensor<1x1x80x128xf16>, tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-NOT:      multiClusterStrategy = #VPU.multi_cluster_strategy<
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PermuteWithoutAlignedInputAndOutputDontAssignedSOK
// CHECK-SAME:     ([[INPUT0:%.+]]: tensor<1x500x16x128xf16>, [[INPUT1:%.+]]: tensor<1x500x16x1xf16>)
func.func @PermuteWithoutAlignedInputAndOutputDontAssignedSOK(%arg0: tensor<1x500x16x128xf16>, %arg1: tensor<1x500x16x1xf16>) -> tensor<1x512x2x128xf16, {order = #NHWC}> {
    %MUL = VPU.Multiply(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x500x16x128xf16>, tensor<1x500x16x1xf16> -> tensor<1x500x16x128xf16>
    %PERMUTE = VPU.NCE.Permute(%MUL) {
        dstElemType = f16,
        dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        expandedChannels = 512 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x512x16x128xf16, {order = #NHWC}>
    %AVG = VPU.NCE.AveragePool(%PERMUTE) {
        kernel_size = [8, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [8, 1]
    } -> tensor<1x512x2x128xf16, {order = #NHWC}>

    //CHECK:        [[MUL:%.+]] = VPU.Multiply([[INPUT0]], [[INPUT1]])
    //CHECK:        [[PERMUTE:%.+]] = VPU.NCE.Permute([[MUL]])
    //CHECK-NOT:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:        [[AVG:%.+]] = VPU.NCE.AveragePool([[PERMUTE]])

    return %AVG : tensor<1x512x2x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PermuteWithInputSWNotCompatibleSOKAssignedSOK
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x384x18x32xf16>
func.func @PermuteWithInputSWNotCompatibleSOKAssignedSOK(%arg0: tensor<1x384x18x32xf16>) -> tensor<1x384x36x64xf16, {order = #NHWC}> {
    %INTERPOLATE = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
            shape_calc_mode = <SIZES>,
            coord_mode = <ALIGN_CORNERS>,
            nearest_mode = <FLOOR>, antialias = false,
            pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
            cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3],
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            scales_attr = [2.000000e+00, 2.000000e+00],
            sizes_attr = [36, 64]
        } : tensor<1x384x18x32xf16> -> tensor<1x384x36x64xf16>

    %PERMUTE = VPU.NCE.Permute(%INTERPOLATE) {dstElemType = f16,
            dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
            expandedChannels = 384 : i64, ppe = #VPU.PPEInt<mode = <NOOP>,
            clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
            lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
        } -> tensor<1x384x36x64xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %PERMUTE : tensor<1x384x36x64xf16, {order = #NHWC}>

    // CHECK:       [[INTERPOLATE:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK:       [[PERMUTE:%.+]] = VPU.NCE.Permute([[INTERPOLATE]]) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 384 : i64,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>,
    // CHECK-SAME:      clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    // CHECK-SAME:      -> tensor<1x384x36x64xf16, {order = #NHWC}>

    //CHECK:        return [[PERMUTE]] : tensor<1x384x36x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOHNotStuck
// CHECK-SAME:      ([[INPUT0:%.+]]: tensor<1x128x28x4608xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOHNotStuck(%arg0: tensor<1x128x28x4608xf16, {order = #NHWC}>) -> tensor<1x128x28x4608xf16, {order = #NWHC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<1x128x28x4608xf16, {order = #NWHC}>

    return %0 : tensor<1x128x28x4608xf16, {order = #NWHC}>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.MaxPool([[INPUT0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x128x28x4608xf16, {order = #NWHC}>

    // CHECK:        return [[VAL0]] : tensor<1x128x28x4608xf16, {order = #NWHC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16, 0.078431372549019607>
!qElemType1 = !quant.uniform<u8:f16, 0.078431372549019607>

config.Resources 6 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

// CHECK-LABEL: func.func @DefaultSOHForSwish
func.func @DefaultSOHForSwish(%arg0: tensor<1x4x640x640x!qElemType0, {order = #NHWC}>) -> tensor<1x16x320x320x!qElemType1, {order = #NHWC}> {
    %cst = const.Declare tensor<16x1x1x32x!qElemType0, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x1x32xf32>, [
            #const.CastElemType<f16>,
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType0>,
            #const.Reorder<#NHWC>
        ]
    %cst1 = const.Declare tensor<16x1x1x4xsi32> =
        dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.CompressConvolution(%arg0, %cst, %cst1) {cm_sp_pattern = 7 : i64, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [16, 3, 3, 3], strides = [2, 2], ppe = #VPU.PPEStub<>} -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %1 = VPU.Swish(%0) {beta_value = 1.000000e+00 : f64} : tensor<1x16x320x320xf16, {order = #NHWC}> -> tensor<1x16x320x320xf16, {order = #NHWC}>
    %2 = VPU.NCE.AveragePool(%1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], ppe = #VPU.PPEStub<>} -> tensor<1x16x320x320x!qElemType1, {order = #NHWC}>
    return %2 : tensor<1x16x320x320x!qElemType1, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.CompressConvolution
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:       [[SWISH:%.+]] = VPU.Swish
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0028915546688379028:131>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DequantizeSmallChannelNotSOK
func.func @DequantizeSmallChannelNotSOK() -> tensor<1x16x512x512xf16, {order = #NHWC}>  {
    %conv_input = const.Declare tensor<1x128x512x512xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.0> : tensor<1x128x512x512xf16>, [#const.Reorder<#NHWC>]
    %weight_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %weights = const.Declare tensor<16x128x3x3x!qElemType, {order = #NHWC}> = dense<1> : tensor<16x128x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %dequantize = VPU.Dequantize(%weights) {dstElemType = f16} : tensor<16x128x3x3x!qElemType, {order = #NHWC}> -> tensor<16x128x3x3xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%conv_input, %dequantize, %weight_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 128, 3, 3], strides = [1, 1]} : tensor<1x128x512x512xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>, tensor<16x128x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x512x512xf16, {order = #NHWC}>

    return %conv :tensor<1x16x512x512xf16, {order = #NHWC}>

    // CHECK:       [[DEQUANT:%.+]] = VPU.Dequantize(
    // CHECK-SAME:  {dstElemType = f16,
    // CHECK-NOT:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    // CHECK:       [[CONV:%.+]]VPU.NCE.Convolution

}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ErrorCodeHeuristicSOK
func.func @ErrorCodeHeuristicSOK(%arg0 : tensor<1x3072x16x4xf16, {order = #NHWC}>) -> tensor<1x4608x16x4xf16, {order = #NHWC}>  {
    %cst = const.Declare tensor<4608x3072x1x1x!qElemType, {order = #NHWC}> = dense<-1.000000e+00> : tensor<4608x3072x1x1xf16>, [#const.CastElemType<ui4>, #const.ConvertElemType<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4608x1x1x4xsi32> = dense<1> : tensor<4608x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%arg0, %cst, %wt) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4608, 3072, 1, 1], strides = [1, 1]} : tensor<1x3072x16x4xf16, {order = #NHWC}>, tensor<4608x3072x1x1x!qElemType, {order = #NHWC}>, tensor<4608x1x1x4xsi32> -> tensor<1x4608x16x4xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %conv :tensor<1x4608x16x4xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]]VPU.NCE.Convolution
    // CHECK:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ErrorCodeHeuristicSOH
func.func @ErrorCodeHeuristicSOH(%arg0 : tensor<1x3072x1152x1xf16, {order = #NHWC}>) -> tensor<1x4608x1152x1xf16, {order = #NHWC}>  {
    %cst = const.Declare tensor<4608x3072x1x1x!qElemType, {order = #NHWC}> = dense<-1.000000e+00> : tensor<4608x3072x1x1xf16>, [#const.CastElemType<ui4>, #const.ConvertElemType<!qElemType>, #const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4608x1x1x4xsi32> = dense<1> : tensor<4608x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%arg0, %cst, %wt) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4608, 3072, 1, 1], strides = [1, 1]} : tensor<1x3072x1152x1xf16, {order = #NHWC}>, tensor<4608x3072x1x1x!qElemType, {order = #NHWC}>, tensor<4608x1x1x4xsi32> -> tensor<1x4608x1152x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    return %conv :tensor<1x4608x1152x1xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]]VPU.NCE.Convolution
    // CHECK:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWith2DimSWParent
func.func @DWConvWith2DimSWParent(%arg0 : tensor<1x768xf16>) -> tensor<1x32x6x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %0 = VPU.NormalizeL2(%arg0) {axes_value = [1], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x768xf16> -> tensor<1x768xf16>
    %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 768, 1, 1]} : tensor<1x768xf16> -> tensor<1x768x1x1xf16>
    %2 = VPU.PermuteCast(%1) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = #NHWC} : tensor<1x768x1x1xf16> -> tensor<1x768x1x1xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 32, 6, 4]} inputs(%2 : tensor<1x768x1x1xf16, {order = #NHWC}>) -> tensor<1x32x6x4xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst, %wt) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x6x4xf16, {order = #NHWC}>

    return %4 : tensor<1x32x6x4xf16, {order = #NHWC}>

    // CHECK:       [[DWCONV:%.+]]VPU.NCE.DepthConvolution
    // CHECK:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOKBasedOnAlignedTile
func.func @ConvAssignedSOKBasedOnAlignedTile(%arg0: tensor<1x128x64x1xf16, {order = #NHWC}>) -> tensor<1x16448x64x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16448x1x1x4xsi32> = dense<10> : tensor<16448x1x1x4xsi32>
    %cst_0 = const.Declare tensor<16448x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16448x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16448, 128, 1, 1], strides = [1, 1]} : tensor<1x128x64x1xf16, {order = #NHWC}>, tensor<16448x128x1x1xf16, {order = #NHWC}>, tensor<16448x1x1x4xsi32> -> tensor<1x16448x64x1xf16, {order = #NHWC}>
    return %0 : tensor<1x16448x64x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<16448x1x1x4xsi32> = dense<10> : tensor<16448x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16448x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16448x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    // CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16448, 128, 1, 1], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x16448x64x1xf16, {order = #NHWC}>

    // CHECK:        return [[VAL0]] : tensor<1x16448x64x1xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0028915546688379028:131>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DefaultDequantizeStrategySOK
func.func @DefaultDequantizeStrategySOK() -> tensor<128x128x10x10xf16, {order = #NHWC}>  {
    %weights = const.Declare tensor<128x128x10x10x!qElemType, {order = #NHWC}> = dense<1> : tensor<128x128x10x10xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %dequantize = VPU.Dequantize(%weights) {dstElemType = f16} : tensor<128x128x10x10x!qElemType, {order = #NHWC}> -> tensor<128x128x10x10xf16, {order = #NHWC}>

    return %dequantize :tensor<128x128x10x10xf16, {order = #NHWC}>

    // CHECK:       [[DEQUANT:%.+]] = VPU.Dequantize(
    // CHECK-SAME:  {dstElemType = f16,
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}

}

// -----

// CHECK-LABEL:   @SDPAAssignedSplitOverKernel
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x32x1x96xf16>, [[ARG1:%.+]]: tensor<1x32x1024x96xf16>, [[ARG2:%.+]]: tensor<1x32x96x1024xf16>, [[ARG3:%.+]]: tensor<1x1x1x1024xf16>)
func.func @SDPAAssignedSplitOverKernel(%arg0: tensor<1x32x1x96xf16>, %arg1: tensor<1x32x1024x96xf16>, %arg2: tensor<1x32x96x1024xf16>, %arg3: tensor<1x1x1x1024xf16>) -> (tensor<1x32x1x96xf16>){
    %0 = VPU.SDPA(%arg0, %arg1, %arg2, %arg3) {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>} : tensor<1x32x1x96xf16>, tensor<1x32x1024x96xf16>, tensor<1x32x96x1024xf16>, tensor<1x1x1x1024xf16> -> tensor<1x32x1x96xf16>
    return %0 : tensor<1x32x1x96xf16>

    // CHECK:       [[SDPA:%.*]] = VPU.SDPA(
    // CHECK-SAME:      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>}

}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: func.func @LSTMSequenceBidirectionalSplitOverKernel(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x2x?x512xf16,
// CHECK-SAME:      -> (tensor<1x2x?x128xf16,
// CHECK-SAME:          tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
func.func @LSTMSequenceBidirectionalSplitOverKernel(%arg0:
            tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 80, 512]> : tensor<4xsi64>}>)
        -> (tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 80, 128]> : tensor<4xsi64>}>,
            tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<1x2x1x128xf16> = dense<1.000000e+00> : tensor<1x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00>
           : tensor<2x4x128x128xf16, {order = #NWHC}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst)
        {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>}
        : tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 80, 512]> : tensor<4xsi64>}>,
          tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>,
          tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x1x1x2xsi32>
        -> tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 80, 128]> : tensor<4xsi64>}>,
           tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState
            : tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 80, 128]> : tensor<4xsi64>}>,
              tensor<1x2x1x128xf16>, tensor<1x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>,
// CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
// CHECK-LABEL: func.func @LSTMSequenceBidirectionalSplitOverBatch(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<3x2x?x512xf16,
// CHECK-SAME:      -> (tensor<3x2x?x128xf16,
// CHECK-SAME:          tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>) {
func.func @LSTMSequenceBidirectionalSplitOverBatch(%arg0:
            tensor<3x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[3, 2, 80, 512]> : tensor<4xsi64>}>)
        -> (tensor<3x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[3, 2, 80, 128]> : tensor<4xsi64>}>,
            tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>) {
    %cst_0 = const.Declare tensor<3x2x1x128xf16> = dense<1.000000e+00> : tensor<3x2x1x128xf16>
    %cst_2 = const.Declare tensor<2x4x128x128xf16, {order = #NWHC}> = dense<3.000000e+00>
           : tensor<2x4x128x128xf16, {order = #NWHC}>
    %cst = const.Declare tensor<1x1x1x2xsi32> = dense<0> : tensor<1x1x1x2xsi32>

    %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%arg0, %cst_0, %cst_0, %cst_2, %cst)
        {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>}
        : tensor<3x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[3, 2, 80, 512]> : tensor<4xsi64>}>,
          tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>,
          tensor<2x4x128x128xf16, {order = #NWHC}>, tensor<1x1x1x2xsi32>
        -> tensor<3x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[3, 2, 80, 128]> : tensor<4xsi64>}>,
           tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>

    return %outputHiddenValues, %outputHiddenState, %outputCellState
            : tensor<3x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[3, 2, 80, 128]> : tensor<4xsi64>}>,
              tensor<3x2x1x128xf16>, tensor<3x2x1x128xf16>

// CHECK:       VPU.LSTMSequence
// CHECK-SAME:      {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>,
// CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>}
}
