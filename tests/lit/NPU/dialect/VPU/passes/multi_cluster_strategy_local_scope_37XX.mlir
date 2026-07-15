//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment="mc-optimization-scope=local" %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>

// CHECK-LABEL: @NoSubgraphOptimization
func.func @NoSubgraphOptimization(%arg0: tensor<1x256x24x16xf16, {order = #NHWC}>, %arg1: tensor<1x256x24x16xf16, {order = #NHWC}>) -> tensor<1x256x24x16x!qElemType, {order = #NHWC}>  {
    %weights = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %add1 = VPU.NCE.Eltwise(%arg0, %arg1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [20688], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [45604], in2_quant_mult = [26278], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x24x16x!qElemType, {order = #NHWC}>
    %conv1 = VPU.NCE.Convolution(%add1, %weights) rawFilterShape [256, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>,
        clamp_low = 0 : i64,
        clamp_high = 255 : i64,
        lrelu_mult = 1 : i64,
        lrelu_shift = 0 : i64,
        fp_prelu_alpha = 1.000000e+00 : f64>,
         strides = [1, 1]
        } : tensor<1x256x24x16x!qElemType, {order = #NHWC}>, tensor<256x256x3x3x!qElemType, {order = #NHWC}> -> tensor<1x256x24x16x!qElemType, {order = #NHWC}>
    %conv2 = VPU.NCE.Convolution(%conv1, %weights) rawFilterShape [256, 256, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x256x24x16x!qElemType, {order = #NHWC}>, tensor<256x256x3x3x!qElemType, {order = #NHWC}> -> tensor<1x256x24x16x!qElemType, {order = #NHWC}>
    %add2 = VPU.NCE.Eltwise(%conv2, %add1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [17510], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [59378], in2_quant_mult = [25950], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x256x24x16x!qElemType, {order = #NHWC}>
    return %add2 : tensor<1x256x24x16x!qElemType, {order = #NHWC}>

    // CHECK: [[ADD1:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK: [[CONV2:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK: [[ADD2:%.+]] = VPU.NCE.Eltwise
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignedAccordingToParentStrategySOK
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x128x2x2xf16, {order = #NHWC}>)
func.func @AssignedAccordingToParentStrategySOK(%arg0: tensor<1x128x2x2xf16, {order = #NHWC}>) -> tensor<1x1024x2x2xf16, {order = #NHWC}> {
    %weightstable = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    %weights = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%arg0, %weights, %weightstable) rawFilterShape [1024, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} : tensor<1x128x2x2xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    %gelu = VPU.Gelu(%conv) : tensor<1x1024x2x2xf16, {order = #NHWC}> -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    return %gelu : tensor<1x1024x2x2xf16, {order = #NHWC}>

    //CHECK-DAG:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<1024x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>
    //CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK:            [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS]], [[WEIGHTSTABLE]]) rawFilterShape [1024, 128, 1, 1]
    //CHECK-SAME:                   {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    //CHECK-SAME:                   -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    //CHECK:            [[GELU:%.+]] = VPU.Gelu([[CONV]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x1024x2x2xf16, {order = #NHWC}> -> tensor<1x1024x2x2xf16, {order = #NHWC}>
    //CHECK:            return [[GELU]] : tensor<1x1024x2x2xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AssignedAccordingToParentStrategySOH
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @AssignedAccordingToParentStrategySOH(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>) -> tensor<1x80x28x28xf16, {order = #NHWC}> {
    %weightstable = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %weights = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%arg0, %weights, %weightstable) rawFilterShape [80, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,  strides = [1, 1]} : tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<80x64x3x3xf16, {order = #NHWC}>, tensor<80x1x1x4xsi32> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    %gelu = VPU.Gelu(%conv) : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    return %gelu : tensor<1x80x28x28xf16, {order = #NHWC}>

    //CHECK-DAG:        [[WEIGHTSTABLE:%.+]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    //CHECK:            [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[WEIGHTS]], [[WEIGHTSTABLE]]) rawFilterShape [80, 64, 3, 3]
    //CHECK-SAME:                {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]}
    //CHECK-SAME:                -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            [[GELU:%.+]] = VPU.Gelu([[CONV]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x80x28x28xf16, {order = #NHWC}> -> tensor<1x80x28x28xf16, {order = #NHWC}>
    //CHECK:            return [[GELU]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}
