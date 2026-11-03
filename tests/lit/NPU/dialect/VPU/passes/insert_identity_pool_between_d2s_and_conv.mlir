//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=NPU5010 compilation-mode=DefaultHW allow-custom-values=true power-optimized=true" --insert-identity-pool-between-d2s-and-conv %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @insertIdentityPool
func.func @insertIdentityPool(%input: tensor<1x64x8x8xf16, {order = #NHWC}>)
        -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %weights3x3 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights5x5 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x5x5xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %d2s = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
        : tensor<1x64x8x8xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %conv3x3 = VPU.NCE.Convolution(%d2s, %weights3x3) rawFilterShape [16, 16, 3, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %conv5x5 = VPU.NCE.Convolution(%d2s, %weights5x5) rawFilterShape [16, 16, 5, 5] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %conv3x3, %conv5x5 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace
    // CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[D2S]])
    // CHECK-SAME:  kernel_size = [1, 1]
    // CHECK-SAME:  strides = [1, 1]
    // CHECK:       VPU.NCE.Convolution([[POOL]], {{%.+}}) rawFilterShape [16, 16, 3, 3]
    // CHECK:       VPU.NCE.Convolution([[POOL]], {{%.+}}) rawFilterShape [16, 16, 5, 5]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @keepOneByOneConv
func.func @keepOneByOneConv(%input: tensor<1x64x8x8xf16, {order = #NHWC}>)
        -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %d2s = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
        : tensor<1x64x8x8xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%d2s, %weights) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %conv : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace
    // CHECK-NOT:   VPU.NCE.AveragePool
    // CHECK:       VPU.NCE.Convolution([[D2S]], {{%.+}}) rawFilterShape [16, 16, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @redirectEarlierOneByOneConv
func.func @redirectEarlierOneByOneConv(%input: tensor<1x64x8x8xf16, {order = #NHWC}>)
        -> (tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>) {
    %weights1x1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weights3x3 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %d2s = VPU.DepthToSpace(%input) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>}
        : tensor<1x64x8x8xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %conv1x1 = VPU.NCE.Convolution(%d2s, %weights1x1) rawFilterShape [16, 16, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %conv3x3 = VPU.NCE.Convolution(%d2s, %weights3x3) rawFilterShape [16, 16, 3, 3] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>
        -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %conv1x1, %conv3x3 : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace
    // CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[D2S]])
    // CHECK-SAME:  kernel_size = [1, 1]
    // CHECK-SAME:  strides = [1, 1]
    // CHECK:       VPU.NCE.Convolution([[POOL]], {{%.+}}) rawFilterShape [16, 16, 1, 1]
    // CHECK:       VPU.NCE.Convolution([[POOL]], {{%.+}}) rawFilterShape [16, 16, 3, 3]
}
