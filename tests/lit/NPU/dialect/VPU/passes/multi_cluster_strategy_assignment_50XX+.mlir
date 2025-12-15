//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvODUAssignedClusteringModule
module @ConvODUAssignedClusteringModule  {

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @ConvODUAssignedClustering
func.func @ConvODUAssignedClustering(%arg0: tensor<1x1024x2x2xf16, {order = #NHWC}>) -> tensor<1x4x2x2xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<4x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4, 1024, 1, 1], strides = [1, 1]} : tensor<1x1024x2x2xf16, {order = #NHWC}>, tensor<4x1024x1x1xf16, {order = #NHWC}> -> tensor<1x4x2x2xf16, {order = #NHWC}>
    return %0 : tensor<1x4x2x2xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<4x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [4, 1024, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x4x2x2xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x4x2x2xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0: tensor<1x64x56x56xf16, {order = #NHWC}>) -> tensor<1x80x56x56xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}> -> tensor<1x80x56x56xf16, {order = #NHWC}>
    return %0 : tensor<1x80x56x56xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:    {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x56x56xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x56x56xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 128, 1, 1], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedClustering
func.func @ConvAssignedClustering(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 64, 3, 3], strides = [1, 1]} : tensor<1x64x1x1xf16, {order = #NHWC}>, tensor<16x64x3x3xf16, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<16x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvAssignedSOH
func.func @DepthConvAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]} -> tensor<1x128x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>, [#const.Reshape<[128, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[128, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 3, 3], strides = [1, 1]}
    //CHECK:        -> tensor<1x128x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x128x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedClustering
func.func @DepthConvAssignedClustering(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 1, 3, 3], strides = [1, 1]} -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]])
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
            strides = [1, 1],
            kernel_size = [1, 1],
            ppe = #VPU.PPEStub<>
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
            strides = [1, 1],
            kernel_size = [1, 1],
            ppe = #VPU.PPEStub<>
         } -> tensor<1x32x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    //CHECK-SAME:   {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]}
    //CHECK-SAME:   -> tensor<1x32x1x1xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolAssignedSOB2Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOB2Batches(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @MaxPoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x112x112xf16, {order = #NHWC}>)
func.func @MaxPoolAssignedSOB(%input: tensor<3x32x112x112xf16, {order = #NHWC}>) -> tensor<3x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>
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

// CHECK-LABEL:  func.func @AvgPoolAssignedSOB2Batches
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<2x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolAssignedSOB2Batches(%input: tensor<2x32x112x112xf16, {order = #NHWC}>) -> tensor<2x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<2x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       [[avgPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<2x32x112x112xf16, {order = #NHWC}>

    // CHECK:       return [[avgPOOL]] : tensor<2x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  func.func @AvgPoolAssignedSOB
// CHECK-SAME:   ([[INPUT:%.+]]: tensor<3x32x112x112xf16, {order = #NHWC}>)
func.func @AvgPoolAssignedSOB(%input: tensor<3x32x112x112xf16, {order = #NHWC}>) -> tensor<3x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>
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

// CHECK-LABEL: @SparseConvAssignedSOH
func.func @SparseConvAssignedSOH(%arg0 : tensor<1x32x28x28xf16, {order = #NHWC}>, %arg1 : tensor<1x32x28x28xi1, {order = #NHWC}>)
        -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {

    %input_sparse = VPU.GroupSparseTensor(%arg0, %arg1)
        -> !VPU.SparseTensor<data=tensor<1x32x28x28xf16, {order = #NHWC}>,
                             sparsity_map=tensor<1x32x28x28xi1, {order = #NHWC}>>

    %weights = const.Declare tensor<80x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<80x1x1x384xi1> = dense<1.000000e+00> : tensor<80x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>, is_weights}
        -> !VPU.SparseTensor<data=tensor<80x32x3x3xf16, {order = #NHWC}>,
                             sparsity_map=tensor<80x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>>


    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 01 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [80, 32, 3, 3],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x32x28x28xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x28x28xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<80x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<80x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>> -> !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                               sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

    return %0 : !VPU.SparseTensor<data=tensor<1x80x28x28xf16, {order = #NHWC}>,
                                  sparsity_map=tensor<1x80x28x28xi1, {order = #NHWC}>>

    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, %arg1)
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x32x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x32x28x28xi1, {order = #NHWC}>>

    // CHECK-DAG:       [[CST_WEIGHTS:%.+]] = const.Declare tensor<80x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    // CHECK-DAG:       [[CST_WEIGHTS_SM:%.+]] = const.Declare tensor<80x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<80x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    // CHECK:           [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[CST_WEIGHTS]], [[CST_WEIGHTS_SM]])
    // CHECK-SAME:      {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<80xi64>, alignment = 16 : i64>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<80x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<80x1x1x384xi1>, is_weights, #VPU.SparsityCompression

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]])
    // CHECK-SAME:          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [80, 32, 3, 3],
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


    %0 = VPU.NCE.Convolution(%input_sparse, %weights_sparse) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [1008, 128, 1, 1],
            strides = [1, 1]
        } : !VPU.SparseTensor<data=tensor<1x128x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x1x1xi1, {order = #NHWC}>>, !VPU.SparseTensor<data=tensor<1008x128x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<1008x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<64> : tensor<1008xi64>, alignment = 16 : i64>> -> !VPU.SparseTensor<data=tensor<1x1008x1x1xf16, {order = #NHWC}>,
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

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[WEIGHTS_SPARSE]])
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

// CHECK-LABEL: @MVNAssignedClusteringOn2
func.func @MVNAssignedClusteringOn2(%arg0: tensor<1x2x1x256xf16, {order = #NCHW}>) -> tensor<1x2x1x256xf16, {order = #NCHW}> {

    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x2x1x256xf16, {order = #NCHW}> -> tensor<1x2x1x256xf16, {order = #NCHW}>

    return %0 : tensor<1x2x1x256xf16, {order = #NCHW}>

    //CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, normalize_variance = true} : tensor<1x2x1x256xf16, {order = #NCHW}> -> tensor<1x2x1x256xf16, {order = #NCHW}>
    //CHECK:   return [[ResultMVN]] : tensor<1x2x1x256xf16, {order = #NCHW}>
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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TanhAfterConvAssignedClustering
func.func @TanhAfterConvAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

    %39 = const.Declare tensor<1x1024x1x1xf16> = dense<10.1> : tensor<1x1024x1x1xf16>
    %40 = const.Declare tensor<1x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1x1xf16, {order = #NHWC}>

    %cst_7 = const.Declare tensor<1024x1024x1x1xf16, {order = #NHWC}> = dense<10.1> : tensor<1x1024x1024xf16>, [#const.Reshape<[1024, 1024]>, #const.Reshape<[1024, 1024, 1, 1]>, #const.Reorder<#NHWC>]

    %42 = VPU.NCE.Convolution(%40, %cst_7) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]} : tensor<1x1024x1x1xf16, {order = #NHWC}>, tensor<1024x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    %45 = VPU.Tanh(%42) : tensor<1x1024x1x1xf16, {order = #NHWC}> -> tensor<1x1024x1x1xf16, {order = #NHWC}>

    return %45 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    //CHECK:   [[ResultConvolution:%.*]] = VPU.NCE.Convolution([[cst0:%.*]], [[cst1:%.*]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [1024, 1024, 1, 1], strides = [1, 1]}
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
    //CHECK-SAME:           {cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
    //CHECK-SAME:           pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
    //CHECK-SAME:           rawFilterShape = [64, 4, 7, 7], strides = [2, 2]}
    //CHECK-SAME:           -> tensor<1x64x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x64x112x112xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOB2Batches
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<2x3x224x224xf16, {order = #NHWC}>)
func.func @CompressConvolutionAssignedSOB2Batches(%arg0: tensor<2x3x224x224xf16, {order = #NHWC}>) -> tensor<2x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<2x3x224x224xf16, {order = #NHWC}> -> tensor<2x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table) {
        cm_sp_pattern = 7 : i64,
        pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 3, 7, 7], strides = [2, 2]
    } -> tensor<2x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<2x64x112x112xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:   [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    // CHECK-SAME:  tensor<2x3x224x224xf16, {order = #NHWC}> -> tensor<2x4x224x224xf16, {order = #NHWC}>

    // CHECK:   [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:   {cm_sp_pattern = 7 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverBatch>,

    // CHECK:    return [[VAL0]] : tensor<2x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOB
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<3x3x224x224xf16, {order = #NHWC}>)
func.func @CompressConvolutionAssignedSOB(%arg0: tensor<3x3x224x224xf16, {order = #NHWC}>) -> tensor<3x64x112x112xf16, {order = #NHWC}> {
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

// Track E#160175 - SEP cost support

// CHECK-LABEL: @InterpolateNearestAssignedSOK
func.func @InterpolateNearestAssignedSOK(%arg0: tensor<1x256x10x10xf16, {order = #NHWC}>) -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x20x20xi1> = dense<1> : tensor<1x256x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [256], dataShape = [1, 256, 10, 10],
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
    // CHECK-SAME:      rawFilterShape = [256, 256, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOK
func.func @InterpolateBilinearAssignedSOK(%arg0: tensor<1x256x10x10xf16, {order = #NHWC}>) -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x256x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x256x21x21xi1> = dense<1> : tensor<1x256x21x21xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [256], dataShape = [1, 256, 10, 10],
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
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        strides = [1, 1],
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
    // CHECK-SAME:      rawFilterShape = [256, 256, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignSOHForLayerWithLargeActivation
func.func @AssignSOHForLayerWithLargeActivation(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>) -> tensor<1x32x250x250xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} : tensor<1x64x250x250xf16, {order = #NHWC}>, tensor<32x64x3x3xf16, {order = #NHWC}> -> tensor<1x32x250x250xf16, {order = #NHWC}>
    return %0 : tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]])
    //CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    //CHECK-SAME:        rawFilterShape = [32, 64, 3, 3], strides = [1, 1]
    //CHECK-SAME:      -> tensor<1x32x250x250xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SoftMaxAssignedSplitOverHeight
func.func @SoftMaxAssignedSplitOverHeight(%arg0: tensor<1x4x8x512xf16, {order = #NHWC}>) -> tensor<1x4x8x512xf16, {order = #NHWC}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 3} : tensor<1x4x8x512xf16, {order = #NHWC}> -> tensor<1x4x8x512xf16, {order = #NHWC}>

    return %1 : tensor<1x4x8x512xf16, {order = #NHWC}>

    //CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x8x512xf16, {order = #NHWC}> -> tensor<1x4x8x512xf16, {order = #NHWC}>
    //CHECK:   return [[ResultSoftMax]] : tensor<1x4x8x512xf16, {order = #NHWC}>
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
    %1 = VPU.NCE.AveragePool(%0) {kernel_size = [1, 1], ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x1x112xf16, {order = #NHWC}>

    %2 = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x32x112xf16, {order = #NHWC}>
    %3 = VPU.NCE.AveragePool(%2) {kernel_size = [1, 1], ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x32x112xf16, {order = #NHWC}>

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

// CHECK-LABEL: @DontAssignStrategy2DConvert
func.func @DontAssignStrategy2DConvert(%arg0: tensor<10x2xf32>) -> tensor<10x2xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<10x2xf32> -> tensor<10x2xf16>
    return %0 : tensor<10x2xf16>
    //CHECK:   [[Result:%.*]] = VPU.Convert([[ARG:%.*]]) {dstElemType = f16} : tensor<10x2xf32> -> tensor<10x2xf16>
    //CHECK:   return [[Result]] : tensor<10x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipSubGraphOptimizationForINFCost
func.func @SkipSubGraphOptimizationForINFCost(%input_data: tensor<1x128x72x72xf16, {order = #NHWC}>, %input_data2: tensor<1x64x144x144xf16, {order = #NHWC}>) -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    // conv
    %convWeights = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%input_data, %convWeights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x72x72xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<1x64x72x72xf16, {order = #NHWC}>

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

    %transposedConv = VPU.NCE.Convolution(%input, %weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]
    } : !VPU.SparseTensor< data=tensor<1x64x72x72xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x147x147xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x147x147xi32, {order = #NHWC}>, #VPU.SEUpsampling< factors = [1, 1], padding = [2, 2, 2, 2]>>, !VPU.SparseTensor<data=tensor<64x64x4x4xf16, {order = #NHWC}>, sparsity_map=tensor<64x1x1x1024xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<16> : tensor<64xi64>, alignment = 16 : i64>> -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // add
    %add = VPU.NCE.Eltwise(%transposedConv, %input_data2) {
        is_inplace = true,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>


    return %add : tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[CONV_WEIGHTS:%.+]] = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x128x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution(%arg0, [[CONV_WEIGHTS]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
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
    // CHECK:        [[DECONV:%.+]] = VPU.NCE.Convolution([[SPARSE_INPUT]], [[DECONV_WEIGHTS]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        rawFilterShape = [64, 64, 4, 4], strides = [1, 1]}
    // CHECK-SAME:        -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        [[ADD:%.+]] = VPU.NCE.Eltwise([[DECONV]], %arg1)
    // CHECK-SAME:        {is_inplace = true,
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:        op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:        } -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:        return [[ADD]] : tensor<1x64x144x144xf16, {order = #NHWC}>
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {
    config.Resources 3 of @NCE at 6.000000e+02 MHz
    // CHECK-LABEL: @MVN1NormalizeAssignedSplitOverKernel
    func.func @MVN1NormalizeAssignedSplitOverKernel(%arg0: tensor<1x3x20x35971xf16>, %arg1: tensor<1x3x1x1xf16, {order = #NHWC}>) -> tensor<1x3x20x35971xf16> {
        %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = false} : tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>
        return %0: tensor<1x3x20x35971xf16>

        //CHECK:        [[VAL0:%.*]] = VPU.MVN1Normalize(%arg0, %arg1)
        // CHECK-SAME:      {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = false} :
        // CHECK-SAME:      tensor<1x3x20x35971xf16>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x20x35971xf16>

        //CHECK:        return [[VAL0]] : tensor<1x3x20x35971xf16>
    }
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module @Test {
    config.Resources 3 of @NCE at 6.000000e+02 MHz
    // CHECK-LABEL: @ConvAssignedSOHWithSOHIncompatibleConsumer
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x256x2000x1xf16, {order = #NHWC}>)
    func.func @ConvAssignedSOHWithSOHIncompatibleConsumer(%arg0: tensor<1x256x2000x1xf16, {order = #NHWC}>) -> tensor<1x256x2000x1xf16, {order = #NHWC}> {

        %weights_0 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.NCE.Convolution(%arg0, %weights_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} : tensor<1x256x2000x1xf16, {order = #NHWC}>, tensor<256x256x3x3xf16, {order = #NHWC}> -> tensor<1x256x2000x1xf16, {order = #NHWC}>

        %weights_1 = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
        %1 = VPU.NCE.Convolution(%0, %weights_1) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 256, 1, 1], strides = [8, 1]} : tensor<1x256x2000x1xf16, {order = #NHWC}>, tensor<64x256x1x1xf16, {order = #NHWC}> -> tensor<1x64x250x1xf16, {order = #NHWC}>

        %2 = VPU.ReduceMean(%0) {axes_value = [2]} : tensor<1x256x2000x1xf16, {order = #NHWC}> -> tensor<1x256x1xf16, {order = #CHW}>

        return %0 : tensor<1x256x2000x1xf16, {order = #NHWC}>

        //CHECK:        [[WEIGHTS0:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]
        //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS0]])
        //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

        //CHECK:        [[WEIGHTS1:%.+]] = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
        //CHECK:        [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]], [[WEIGHTS1]])
        //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

        //CHECK:        [[VAL1:%.+]] = VPU.ReduceMean([[VAL0]])
        //CHECK-NOT:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

        //CHECK:        return [[VAL0]] : tensor<1x256x2000x1xf16, {order = #NHWC}>
    }
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @GroupedMatMulAssignedSOG
func.func @GroupedMatMulAssignedSOG(%arg0:  tensor<4x1x32x64x1xf16, {order =#GNHWC}>, %arg1: tensor<4x64x32x1x1xf16, {order = #GNHWC}>) -> tensor<4x1x64x64x1xf16, {order = #GNHWC}> {
    %0 = VPU.NCE.MatMul(%arg0, %arg1) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [4, 1, 64, 32, 1], strides = [1, 1]
    } -> tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    return %0 : tensor<4x1x64x64x1xf16, {order = #GNHWC}>

    // CHECK:        [[MATMUL:%.+]] = VPU.NCE.MatMul
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>
    // CHECK:        return [[MATMUL]] : tensor<4x1x64x64x1xf16, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @module  {

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSUMAssignedSOH
func.func @NCEReduceSUMAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

}
// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

// -----

config.Resources 4 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

!qElemType = !quant.uniform<u8<0:254>:f16:0, {0.001357270859357879:127,0.0016182628203564742:127,0.0020815957718946804:127,0.0026390524361077257:127,0.0010264177960673654:127,0.0028241668629834034:127,9.4622608244888429E-4:127,0.0012354281708950132:127,0.0010431431175216914:127,0.0020313619628665952:127,0.0023819006334139608:127,0.0036940675551497093:127,0.0040705110144427445:127,0.0010953865886673213:127,0.0024318298486274057:127,0.0024014103600359339:127,0.0039203847487141764:127,0.0018524924131828968:127,0.0026850329609367793:127,0.002006289292508223:127,0.0038565505207992915:127,0.0011652208219362994:127,0.0015601711714361597:127,0.0016786359895871381:127,0.0030253930824009454:127,0.0033942524373062012:127,0.0021159609002391186:127,0.0027468558840864285:127,0.003248042243672168:127,8.3886902398011813E-4:127,0.001087947741268188:127,0.0034467451215728999:127,0.0012410967134115264:127,0.0016006742875407062:127,8.0011086905096458E-4:127,0.0012080863000839715:127,6.9277856762953627E-4:127,0.0010478224341324933:127,0.0013355103534037672:127,0.0016338377017674484:127,0.0040408232080654839:127,0.0022144775221666952:127,0.0012906775699825738:127,0.0031165220136717547:127,0.0023295567260952447:127,0.0038008764972836955:127,0.0041627433356337656:127,0.0025403243819559652:127,0.0040322837867136077:127,7.5960124102164442E-4:127,0.0035432028019522117:127,0.0027394897825135959:127,0.0023073042471577804:127,0.0036451314377972462:127,0.0019629199908474297:127,0.0037665498538280097:127,7.3536362235001695E-4:127,0.0019541361669855794:127,0.0039462034157880651:127,0.0012378619881126824:127,9.717483689465861E-4:127,0.0013865374439344632:127,0.0026975860745888057:127,0.0011498732125665258:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.001357270859357879,0.0016182628203564742,0.0020815957718946804,0.0026390524361077257,0.0010264177960673654,0.0028241668629834034,9.4622608244888429E-4,0.0012354281708950132,0.0010431431175216914,0.0020313619628665952,0.0023819006334139608,0.0036940675551497093,0.0040705110144427445,0.0010953865886673213,0.0024318298486274057,0.0024014103600359339,0.0039203847487141764,0.0018524924131828968,0.0026850329609367793,0.002006289292508223,0.0038565505207992915,0.0011652208219362994,0.0015601711714361597,0.0016786359895871381,0.0030253930824009454,0.0033942524373062012,0.0021159609002391186,0.0027468558840864285,0.003248042243672168,8.3886902398011813E-4,0.001087947741268188,0.0034467451215728999,0.0012410967134115264,0.0016006742875407062,8.0011086905096458E-4,0.0012080863000839715,6.9277856762953627E-4,0.0010478224341324933,0.0013355103534037672,0.0016338377017674484,0.0040408232080654839,0.0022144775221666952,0.0012906775699825738,0.0031165220136717547,0.0023295567260952447,0.0038008764972836955,0.0041627433356337656,0.0025403243819559652,0.0040322837867136077,7.5960124102164442E-4,0.0035432028019522117,0.0027394897825135959,0.0023073042471577804,0.0036451314377972462,0.0019629199908474297,0.0037665498538280097,7.3536362235001695E-4,0.0019541361669855794,0.0039462034157880651,0.0012378619881126824,9.717483689465861E-4,0.0013865374439344632,0.0026975860745888057,0.0011498732125665258}>
!qElemType3 = !quant.uniform<u8:f16, 0.019849412581499887>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvBeforeMVNAssignedSplitOverKernel
func.func @ConvBeforeMVNAssignedSplitOverKernel(%arg0: tensor<1x32x64x128x!qElemType3, {order = #NHWC}>) -> tensor<1x64x64x128xf16> {
    %cst_0 = const.Declare tensor<64x32x4x4x!qElemType, {order = #NHWC}> = dense<10.1> : tensor<64x32x4x4xf32>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %4 = VPU.NCE.Convolution(%arg0, %cst_0) {pad = #VPU.Padding<left = 2 : i64, right = 1 : i64, top = 2 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 32, 4, 4], strides = [1, 1]} : tensor<1x32x64x128x!qElemType3, {order = #NHWC}>, tensor<64x32x4x4x!qElemType, {order = #NHWC}> -> tensor<1x64x64x128xf16>
    %5 = VPU.MVN(%4) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x64x64x128xf16> -> tensor<1x64x64x128xf16>
    return %5 : tensor<1x64x64x128xf16>

    //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution({{[^:]+}}, {{[^:]+}}) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK:    [[MVN:%.+]] = VPU.MVN([[CONV]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:    return [[MVN]] : tensor<1x64x64x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvODUAutopad
module @DepthConvODUAutopad {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %nce = VPU.NCE.DepthConvolution(%input, %weights) {
        input_padding = [0, 8, 0, 0],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [8, 1, 3, 3],
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolODUAutopad
module @MaxPoolODUAutopad {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.MaxPool(%input) {
        input_padding = [0, 8, 0, 0],
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.MaxPool
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolODUAutopad
module @AveragePoolODUAutopad {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.AveragePool(%input) {
        input_padding = [0, 8, 0, 0],
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.AveragePool
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseODUAutopad
module @EltwiseODUAutopad {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.Eltwise(%input, %input) {
        input_padding = [0, 8, 0, 0],
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // CHECK:       VPU.NCE.Eltwise
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @module  {

config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanAssignedSOH
func.func @NCEReduceMeanAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

}
// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

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
