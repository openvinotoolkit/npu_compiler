//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --wrap-in-vertical-fusion %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapNCETiledTask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,

                strides = [1, 1],
                tilingStrategy = [1, 1, 2, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ({{[^:]+}} as [[INPUT:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>, {{[^:]+}} as [[WEIGHTS:%.+]]: tensor<32x32x3x3xf16, {order = #NHWC}>)
    // CHECK-SAME:  attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    // CHECK:  VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]]) rawFilterShape [32, 32, 3, 3] {
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:  strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @WrapNCENonTiledTask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,

                strides = [1, 1]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ({{[^:]+}} as [[INPUT:%.+]]: tensor<1x32x256x256xf16, {order = #NHWC}>, {{[^:]+}} as [[WEIGHTS:%.+]]: tensor<32x32x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:  attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    // CHECK:  VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]]) rawFilterShape [32, 32, 1, 1] {
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME: strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield

}

// -----

// CHECK-LABEL: @WrapActivation
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x3x512x512xf16>)
func.func @WrapActivation(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x3x512x512xf16> {
    %0 = VPU.Tanh(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    return %0 : tensor<1x3x512x512xf16>

    // CHECK:  VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x3x512x512xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x3x512x512xf16> {
    // CHECK:  VPU.Tanh([[ARG_1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK-SAME:  tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    // CHECK:    VPU.Yield

}

// -----

// CHECK-LABEL: @WrapSwish
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x176x176xf16>)
func.func @WrapSwish(%arg0: tensor<1x32x176x176xf16>) -> tensor<1x32x176x176xf16> {
    %0 = VPU.Swish(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x32x176x176xf16> -> tensor<1x32x176x176xf16>
    return %0 : tensor<1x32x176x176xf16>
    // CHECK:  VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x32x176x176xf16>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x176x176xf16> {
    // CHECK:  VPU.Swish([[ARG_1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  tensor<1x32x176x176xf16> -> tensor<1x32x176x176xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @WrapDepthToSpaceBF
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @WrapDepthToSpaceBF(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %2 = VPU.DepthToSpace(%arg0) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 1, 1]} : tensor<1x64x128x128x!qElemType, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %2 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    // CHECK:  VPU.DepthToSpace([[INNER_INPUT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x64x128x128x!qElemType, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapDepthToSpaceDF
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x180x270xf16, {order = #NHWC}>
func.func @WrapDepthToSpaceDF(%arg0: tensor<1x128x180x270xf16, {order = #NHWC}>) -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 15, 1]} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x8x720x1080xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_INPUT:%.+]]: tensor<1x128x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    // CHECK:  VPU.DepthToSpace([[INNER_INPUT]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapMultiply
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>)
func.func @WrapMultiply(%arg0: tensor<1x4x720x1080xf16, {order = #NHWC}>, %arg1: tensor<1x4x720x1080xf16, {order = #NHWC}>) -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 5, 1]} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x4x720x1080xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[ARG_0]] as [[ARG_2:%[^:]+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[ARG_1]] as [[ARG_3:%[^:]+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    // CHECK:  VPU.Multiply([[ARG_2]], [[ARG_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapSubtract
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>)
func.func @WrapSubtract(%arg0: tensor<1x4x720x1080xf16, {order = #NHWC}>, %arg1: tensor<1x4x720x1080xf16, {order = #NHWC}>) -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    %0 = VPU.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 5, 1]} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    return %0 : tensor<1x4x720x1080xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>, [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x4x720x1080xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x4x720x1080xf16, {order = #NHWC}> {
    // CHECK:  VPU.Subtract([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x720x1080xf16, {order = #NHWC}>, tensor<1x4x720x1080xf16, {order = #NHWC}> -> tensor<1x4x720x1080xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapTiledMVN
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x192x16x48xf16, {order = #NHWC}>)
func.func @WrapTiledMVN(%arg0: tensor<1x192x16x48xf16, {order = #NHWC}>) -> tensor<1x192x16x48xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true, tilingStrategy = [1, 2, 1, 1]} : tensor<1x192x16x48xf16, {order = #NHWC}> -> tensor<1x192x16x48xf16, {order = #NHWC}>
    return %0 : tensor<1x192x16x48xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x192x16x48xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x192x16x48xf16, {order = #NHWC}> {
    // CHECK:  VPU.MVN([[ARG_1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x192x16x48xf16, {order = #NHWC}> -> tensor<1x192x16x48xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapGelu
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x128x128xf16, {order = #NHWC}>
func.func @WrapGelu(%arg0: tensor<1x64x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %0 = VPU.Gelu(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x64x128x128xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    // CHECK:  VPU.Gelu([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapSigmoid
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x64x128x128xf16, {order = #NHWC}>
func.func @WrapSigmoid(%arg0: tensor<1x64x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %0 = VPU.Sigmoid(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    return %0 : tensor<1x64x128x128xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    // CHECK:  VPU.Sigmoid([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DontWrapMultiDimTiledNCETask(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>, %weights: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
       %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,

                strides = [1, 1],
                tilingStrategy = [1, 1, 2, 4]} : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
    return %0 : tensor<1x32x256x256xf16, {order = #NHWC}>

    // CHECK:  VPU.NCE.Convolution([[ARG_0:%.+]], [[ARG_1:%.+]]) rawFilterShape [32, 32, 3, 3] {
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME: strides = [1, 1], tilingStrategy = [1, 1, 2, 4]}
    // CHECK-SAME:  -> tensor<1x32x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapAbs
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapAbs(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.Abs(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:  VPU.Abs([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapPRelu
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
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

// -----

// CHECK-LABEL: @WrapSWAdd
// CHECK-SAME: [[INPUT0:%.+]]: tensor<1x32x1024x1024xf16>, [[INPUT1:%.+]]: tensor<1x1x1024x1024xf16>
func.func @WrapSWAdd(%arg0: tensor<1x32x1024x1024xf16>, %arg1: tensor<1x1x1024x1024xf16>) -> tensor<1x32x1024x1024xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 49, 1]} : tensor<1x32x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x32x1024x1024xf16>
    return %0 : tensor<1x32x1024x1024xf16>

    // CHECK: VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x1024x1024xf16>, [[INPUT1]] as [[INNER_ARG1:[^:]+]]: tensor<1x1x1024x1024xf16>) attributes {tilingStrategy = [1, 1, 49, 1]} -> tensor<1x32x1024x1024xf16>
    // CHECK: VPU.Add([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x32x1024x1024xf16>
    // CHECK: VPU.Yield
}

// -----

// CHECK-LABEL: @WrapSWSqrt
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1024x1x14336xf16>
func.func @WrapSWSqrt(%arg0: tensor<1x1024x1x14336xf16>) -> tensor<1x1024x1x14336xf16> {
    %0 = VPU.Sqrt(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 16, 1, 1]} : tensor<1x1024x1x14336xf16> -> tensor<1x1024x1x14336xf16>
    return %0 : tensor<1x1024x1x14336xf16>

    // CHECK: VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG:[^:]+]]: tensor<1x1024x1x14336xf16>) attributes {tilingStrategy = [1, 16, 1, 1]} -> tensor<1x1024x1x14336xf16>
    // CHECK: VPU.Sqrt([[INNER_ARG]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x1x14336xf16> -> tensor<1x1024x1x14336xf16>
    // CHECK: VPU.Yield
}

// -----

// CHECK-LABEL: @WrapSWNormalizeL2
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x60x801x384xf16>
func.func @WrapSWNormalizeL2(%arg0: tensor<1x60x801x384xf16>) -> tensor<1x60x801x384xf16> {
    %0 = VPU.NormalizeL2(%arg0) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 13, 1]} : tensor<1x60x801x384xf16> -> tensor<1x60x801x384xf16>

    return %0 : tensor<1x60x801x384xf16>

    // CHECK: VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG:[^:]+]]: tensor<1x60x801x384xf16>) attributes {tilingStrategy = [1, 1, 13, 1]} -> tensor<1x60x801x384xf16>
    // CHECK: VPU.NormalizeL2([[INNER_ARG]]) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x60x801x384xf16> -> tensor<1x60x801x384xf16>
    // CHECK: VPU.Yield
}

// -----

!qElemType = !quant.uniform<u2:f32, 1.000000e+00>

// CHECK-LABEL: @WrapSWDynamicDequantize
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x3072x48x64x!qElemType>, [[INPUT1:%.+]]: tensor<1x3072x48x1xf16>, [[INPUT2:%.+]]: tensor<1x3072x48x1xui2>)
func.func @WrapSWDynamicDequantize(%arg0: tensor<1x3072x48x64x!qElemType>, %arg1: tensor<1x3072x48x1xf16>, %arg2: tensor<1x3072x48x1xui2>) -> tensor<1x3072x48x64xf16> {
    %0 = VPU.DynamicDequantize(%arg0, %arg1, %arg2) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 4, 1, 1]} : tensor<1x3072x48x64x!qElemType>, tensor<1x3072x48x1xf16>, tensor<1x3072x48x1xui2> -> tensor<1x3072x48x64xf16>
    return %0 :  tensor<1x3072x48x64xf16>

    // CHECK: VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:[^:]+]]: tensor<1x3072x48x64x!qElemType>, [[INPUT1]] as [[INNER_ARG1:[^:]+]]: tensor<1x3072x48x1xf16>, [[INPUT2]] as [[INNER_ARG2:[^:]+]]: tensor<1x3072x48x1xui2>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x3072x48x64xf16> {
    // CHECK: VPU.DynamicDequantize([[INNER_ARG0]], [[INNER_ARG1]], [[INNER_ARG2]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x3072x48x64x!qElemType>, tensor<1x3072x48x1xf16>, tensor<1x3072x48x1xui2> -> tensor<1x3072x48x64xf16>
    // CHECK: VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapRelu
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapRelu(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.ReLU(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:    VPU.ReLU([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapSin
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapSin(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.Sin(%arg0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:    VPU.Sin([[INNER_ARG1]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @WrapNCEMatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<256x1x64x16x16xf16, {order = #GNHWC}>
func.func @WrapNCEMatMul(%arg0: tensor<256x1x64x16x16xf16, {order = #GNHWC}>) -> tensor<256x1x128x16x16xf16, {order = #GNHWC}> {
    %weights = const.Declare tensor<256x128x64x1x1xf16, {order = #GNHWC}> = dense<1.0> : tensor<256x128x64x1x1xf16>, [#const.Reorder<#GNHWC>]
    %0 = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [256, 128, 64, 1, 1]
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1],
            tilingStrategy = [2, 1, 1, 1, 1],
            resultSegmentSizes = array<i32: 1, 0, 0, 0>
    } -> tensor<256x1x128x16x16xf16, {order = #GNHWC}>
    return %0 : tensor<256x1x128x16x16xf16, {order = #GNHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<256x128x64x1x1xf16, {order = #GNHWC}>
    // CHECK:       [[VF_OUTPUT:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_INPUT:[^:]+]]: tensor<256x1x64x16x16xf16, {order = #GNHWC}>, [[WEIGHTS]] as [[INNER_WEIGHTS:[^:]+]]: tensor<256x128x64x1x1xf16, {order = #GNHWC}>) attributes {tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<256x1x128x16x16xf16, {order = #GNHWC}> {
    // CHECK:         [[MM_OUTPUT:%.+]] = VPU.NCE.MatMul([[INNER_INPUT]], [[INNER_WEIGHTS]]) rawFilterShape [256, 128, 64, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<256x1x128x16x16xf16, {order = #GNHWC}>
    // CHECK:         VPU.Yield [[MM_OUTPUT]]
    // CHECK:       }
    // CHECK:       return [[VF_OUTPUT]] : tensor<256x1x128x16x16xf16, {order = #GNHWC}>
}

// -----

// CHECK-LABEL: @WrapRMS
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x128x1x256xf16>)
func.func @WrapRMS(%arg0: tensor<1x128x1x256xf16>) -> tensor<1x128x1x256xf16> {
    %cst = const.Declare tensor<1x1x1x256xf16> = dense<1.0> : tensor<1x1x1x256xf16>
    %0 = VPU.RMS(%arg0, %cst) {eps = 1.0132789611816406E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 2, 1, 1]} : tensor<1x128x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x128x1x256xf16>
    return %0 : tensor<1x128x1x256xf16>

    // CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<1x1x1x256xf16> = dense<1.000000e+00> : tensor<1x1x1x256xf16>
    // CHECK:        VPU.VerticalFusion ([[ARG_0]] as [[INNER_ARG0:[^:]+]]: tensor<1x128x1x256xf16>, [[CST]] as [[INNER_ARG1:[^:]+]]: tensor<1x1x1x256xf16>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x1x256xf16> {
    // CHECK:            VPU.RMS([[INNER_ARG0]], [[INNER_ARG1]]) {eps = 1.0132789611816406E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x128x1x256xf16>
    // CHECK:            VPU.Yield
}

// -----

// CHECK-LABEL: @WrapBitwiseAnd
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x1024x2048xsi32>, [[ARG_1:%[^:]+]]: tensor<1x1x1024x2048xsi32>)
func.func @WrapBitwiseAnd(%arg0: tensor<1x1x1024x2048xsi32>, %arg1: tensor<1x1x1024x2048xsi32>) -> tensor<1x1x1024x2048xsi32> {
    %0 = VPU.BitwiseAnd(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x1x1024x2048xsi32>, tensor<1x1x1024x2048xsi32> -> tensor<1x1x1024x2048xsi32>
    return %0 : tensor<1x1x1024x2048xsi32>

    // CHECK: VPU.VerticalFusion ([[ARG_0]] as [[INNER_ARG0:[^:]+]]: tensor<1x1x1024x2048xsi32>, [[ARG_1]] as [[INNER_ARG1:[^:]+]]: tensor<1x1x1024x2048xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x1x1024x2048xsi32>
    // CHECK: VPU.BitwiseAnd([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x1024x2048xsi32>, tensor<1x1x1024x2048xsi32> -> tensor<1x1x1024x2048xsi32>
    // CHECK: VPU.Yield
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @WrapRound
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x448x392xf16, {order = #NHWC}>
func.func @WrapRound(%arg0: tensor<1x16x448x392xf16, {order = #NHWC}>) -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    return %0 : tensor<1x16x448x392xf16, {order = #NHWC}>

    // CHECK:  VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x448x392xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x448x392xf16, {order = #NHWC}> {
    // CHECK:    VPU.Round([[INNER_ARG1]]) {mode = #IE.round_mode<HALF_TO_EVEN>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x448x392xf16, {order = #NHWC}> -> tensor<1x16x448x392xf16, {order = #NHWC}>
    // CHECK:    VPU.Yield
}
