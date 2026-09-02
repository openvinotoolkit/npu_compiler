//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvKeepsSOHWhenOutputSlicedOnNonHighestDim
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x180x160xf16, {order = #NHWC}>
func.func @ConvKeepsSOHWhenOutputSlicedOnNonHighestDim(%arg0: tensor<1x16x180x160xf16, {order = #NHWC}>) -> tensor<1x32x90x160xf16, {order = #NHWC}> {
    %wt_conv = const.Declare tensor<16x16x2x1xf16, {order = #NHWC}> = dense<1.0e+00> : tensor<16x16x2x1xf16>, [#const.Reorder<#NHWC>]
    %bias_conv = const.Declare tensor<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>
    %wt_compress = const.Declare tensor<32x1x1x48xf16, {order = #NHWC}> = dense<1.0e+00> : tensor<32x1x1x48xf16>, [#const.Reorder<#NHWC>]
    %bias_compress = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%arg0, %wt_conv, %bias_conv) rawFilterShape [16, 16, 2, 1] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        strides = [2, 1]
    } : tensor<1x16x180x160xf16, {order = #NHWC}>, tensor<16x16x2x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32>
      -> tensor<1x16x90x160xf16, {order = #NHWC}>

    %slice = VPU.Slice %conv [0, 0, 0, 0] [1, 4, 90, 160]
        : tensor<1x16x90x160xf16, {order = #NHWC}> to tensor<1x4x90x160xf16, {order = #NHWC}>

    %compress = VPU.NCE.CompressConvolution(%slice, %wt_compress, %bias_compress) rawFilterShape [32, 4, 3, 3] {
        cm_sp_pattern = 15 : i64,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
    } : tensor<1x4x90x160xf16, {order = #NHWC}>, tensor<32x1x1x48xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32>
      -> tensor<1x32x90x160xf16, {order = #NHWC}>

    return %compress : tensor<1x32x90x160xf16, {order = #NHWC}>

    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]]
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK: [[SLICE:%.+]] = VPU.Slice [[CONV]]

    // CHECK: [[COMPRESS:%.+]] = VPU.NCE.CompressConvolution([[SLICE]]
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>

    // CHECK: return [[COMPRESS]]
}
