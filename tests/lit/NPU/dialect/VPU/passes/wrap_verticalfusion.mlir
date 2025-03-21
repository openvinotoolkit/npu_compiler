//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --wrap-in-vertical-fusion %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapNCETiledTask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %wt: tensor<32x1x1x4xsi32>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights, %wt)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [32, 32, 3, 3],
                strides = [1, 1],
                tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg3: tensor<1x32x256x256xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<32x32x3x3xf16, {order = #NHWC}>, %arg1 as %arg5: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:  attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    //CHECK:  VPU.NCE.Convolution(%arg3, %arg4, %arg5)
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapNCENonTiledTask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %wt: tensor<32x1x1x4xsi32>, %weights: tensor<32x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights, %wt)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [32, 32, 1, 1],
                strides = [1, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg3: tensor<1x32x256x256xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<32x32x1x1xf16, {order = #NHWC}>, %arg1 as %arg5: tensor<32x1x1x4xsi32>)
    //CHECK-SAME:  attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    //CHECK:  VPU.NCE.Convolution(%arg3, %arg4, %arg5)
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:  rawFilterShape = [32, 32, 1, 1], strides = [1, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield

}

// -----

func.func @WrapActivation(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x3x512x512xf16> {
    %0 = VPU.Tanh(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    return %0 : tensor<1x3x512x512xf16>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg1: tensor<1x3x512x512xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x3x512x512xf16> {
    //CHECK:  VPU.Tanh(%arg1)
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    //CHECK-SAME:  tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    //CHECK:    VPU.Yield

}

// -----

func.func @WrapSwish(%arg0: tensor<1x32x176x176xf16>) -> tensor<1x32x176x176xf16> {
    %0 = VPU.Swish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x32x176x176xf16> -> tensor<1x32x176x176xf16>
    return %0 : tensor<1x32x176x176xf16>
    //CHECK:  VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x176x176xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x176x176xf16> {
    //CHECK:  VPU.Swish(%arg1)
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:  tensor<1x32x176x176xf16> -> tensor<1x32x176x176xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapDepthToSapce(%arg0: tensor<1x128x180x270xf16, {order = #NHWC}>) -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 15, 1]} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x8x720x1080xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    //CHECK:  VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapMultiply(%arg0: tensor<1x4x720x1080xf16, {order = #NHWC}>, %arg1: tensor<1x4x720x1080xf16, {order = #NHWC}>) -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 5, 1]} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x4x720x1080xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x720x1080xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x720x1080xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    //CHECK:  VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @WrapSubtract
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>)
func.func @WrapSubtract(%arg0: tensor<1x4x720x1080xf16, {order = #NHWC}>, %arg1: tensor<1x4x720x1080xf16, {order = #NHWC}>) -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 5, 1]} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x4x720x1080xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    //CHECK:  VPU.Subtract([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapTiledMVN(%arg0: tensor<1x192x16x48xf16, {order = #NHWC}>) -> tensor<1x192x16x48xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, tilingStrategy = [1, 2, 1, 1]} : tensor<1x192x16x48xf16, {order = #NHWC}> -> tensor<1x192x16x48xf16, {order = #NHWC}>
    return %0 : tensor<1x192x16x48xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion (%arg0 as %arg1: tensor<1x192x16x48xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x192x16x48xf16, {order = #NHWC}> {
    //CHECK:  VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x192x16x48xf16, {order = #NHWC}> -> tensor<1x192x16x48xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @WrapGelu
//CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x128x128xf16, {order = #NHWC}>
func.func @WrapGelu(%arg0: tensor<1x64x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x64x128x128xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    //CHECK:  VPU.Gelu([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @WrapSigmoid
//CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x128x128xf16, {order = #NHWC}>
func.func @WrapSigmoid(%arg0: tensor<1x64x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %0 = VPU.Sigmoid(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x64x128x128xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    //CHECK:  VPU.Sigmoid([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DontWrapMultiDimTiledNCETask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %wt: tensor<32x1x1x4xsi32>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights, %wt)
                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [32, 32, 3, 3],
                strides = [1, 1],
                tilingStrategy = [1, 1, 2, 4]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    //CHECK:  VPU.NCE.Convolution([[ARG_0:%.*]], [[ARG_1:%.*]], [[ARG_2:%.*]])
    //CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 4]} -> tensor<1x32x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @WrapAbs
//CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapAbs(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.Abs(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    //CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    //CHECK:  VPU.Abs([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    //CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @WrapPRelu
//CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapPRelu(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
    %0 = VPU.PRelu(%arg0, %cst) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare
    // CHECK-SAME:       tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]
    // CHECK:        VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>, [[CST]] as [[INNER_ARG2:[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:            VPU.PRelu([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    // CHECK:            VPU.Yield
}
