//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ValidateMCStrategies(%arg0: tensor<1x64x208x208xf16, {order = #NHWC}>) -> tensor<1x64x208x208xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<0> : tensor<64x1x1x4xsi32>

    // expected-error@+1 {{Operations in the block have different MC strategies}}
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x208x208xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x64x1x1xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>, %cst_2 as %arg4: tensor<64x32x3x3xf16, {order = #NHWC}>, %cst_3 as %arg5: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.0999755859375 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64>, rawFilterShape = [32, 64, 1, 1], strides = [1, 1]} -> tensor<1x32x208x208xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg5) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.0999755859375 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %0 : tensor<1x64x208x208xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ValidateWithoutMCStrategies(%arg0: tensor<1x64x208x208xf16, {order = #NHWC}>) -> tensor<1x64x208x208xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<0> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<0> : tensor<64x1x1x4xsi32>

    // expected-error@+1 {{Operations in the block have different MC strategies}}
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x208x208xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x64x1x1xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>, %cst_2 as %arg4: tensor<64x32x3x3xf16, {order = #NHWC}>, %cst_3 as %arg5: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.0999755859375 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64>, rawFilterShape = [32, 64, 1, 1], strides = [1, 1]} -> tensor<1x32x208x208xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg5) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPETask<mode = <LPRELU>, clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 0.0999755859375 : f64, lrelu_mult = 1638 : i64, lrelu_shift = 14 : i64>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %0 : tensor<1x64x208x208xf16, {order = #NHWC}>
}

// -----

func.func @Invalid_Yield_Only_VF_op(%arg0: tensor<1x64x300x300xf32>) -> tensor<1x64x300x300xf32> {
    // expected-error @below {{Operation does not have any operation besides YieldOp}}
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x300x300xf32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x300x300xf32> {
      VPU.Yield %arg1
    }

    return %0 : tensor<1x64x300x300xf32>
}
