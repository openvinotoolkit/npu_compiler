//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=HostCompile allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertSameStrategyStaticAndDynamic
func.func @ConvertSameStrategyStaticAndDynamic(%arg0: tensor<1x3x1600x2560xf32>, %arg1: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x3x1600x2560xf16>, tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x3x1600x2560xf32>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1600x2560xf32> -> tensor<1x3x1600x2560xf16>
    %1 = VPU.Convert(%arg1) {dstElemType = f16} : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    return %0, %1 : tensor<1x3x1600x2560xf16>, tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert([[ARG0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[ARG1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:       return [[CONVERT0]], [[CONVERT1]] : tensor<1x3x1600x2560xf16>, tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteSameStrategyStaticAndDynamic
func.func @PermuteSameStrategyStaticAndDynamic(%arg0: tensor<1x16x1600x2560xf16>, %arg1: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>)
    -> (tensor<1x16x1600x2560xf16, {order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x1600x2560xf16>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}>
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NCHW, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} : tensor<1x16x1600x2560xf16> -> tensor<1x16x1600x2560xf16, {order = #NHWC}>
    %1 = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NCHW, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    return %0, %1 : tensor<1x16x1600x2560xf16, {order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       [[PERMUTE0:%.+]] = VPU.NCE.Permute([[ARG0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:       [[PERMUTE1:%.+]] = VPU.NCE.Permute([[ARG1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:       return [[PERMUTE0]], [[PERMUTE1]] : tensor<1x16x1600x2560xf16, {order = #NHWC}>, tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionSameStrategyStaticAndDynamic
func.func @ConvolutionSameStrategyStaticAndDynamic(%arg0: tensor<1x32x800x1280xf16, {order = #NHWC}>, %arg1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>)
    -> (tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16, {order = #NHWC}>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16, {order = #NHWC}>
    %0 = VPU.NCE.Convolution(%arg0, %cst_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x800x1280xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%arg1, %cst_0) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %0, %1 : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST_0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG1]], [[CST_0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:       return [[CONV0]], [[CONV1]] : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvolutionSameStrategyStaticAndDynamic
func.func @DepthConvolutionSameStrategyStaticAndDynamic(%arg0: tensor<1x32x800x1280xf16, {order = #NHWC}>, %arg1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>)
    -> (tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST_0:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16, {order = #NHWC}>
    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_0) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x800x1280xf16, {order = #NHWC}>
    %1 = VPU.NCE.DepthConvolution(%arg1, %cst_0) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %0, %1 : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       [[DEPTH_CONV0:%.+]] = VPU.NCE.DepthConvolution([[ARG0]], [[CST_0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:       [[DEPTH_CONV1:%.+]] = VPU.NCE.DepthConvolution([[ARG1]], [[CST_0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:       return [[DEPTH_CONV0]], [[DEPTH_CONV1]] : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseSameStrategyStaticAndDynamic
func.func @EltwiseSameStrategyStaticAndDynamic(%arg0: tensor<1x32x800x1280xf16, {order = #NHWC}>, %arg1: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>)
    -> (tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x32x800x1280xf16, {order = #NHWC}>
    // CHECK-SAME: [[ARG1:%.+]]: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x800x1280xf16, {order = #NHWC}> -> tensor<1x32x800x1280xf16, {order = #NHWC}>
    %1 = VPU.NCE.Eltwise(%arg1, %arg1) {is_inplace = true, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %0, %1 : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:        [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG0]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:        [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ARG1]], [[ARG1]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:        return [[ELTWISE0]], [[ELTWISE1]] : tensor<1x32x800x1280xf16, {order = #NHWC}>, tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 800, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolSameStrategyStaticAndDynamic
func.func @MaxPoolSameStrategyStaticAndDynamic(%arg0: tensor<1x16x720x1280xf16, {order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>)
    -> (tensor<1x16x720x1280xf16, {order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x720x1280xf16, {order = #NHWC}>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %0 = VPU.NCE.MaxPool(%arg0) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x720x1280xf16, {order = #NHWC}> -> tensor<1x16x720x1280xf16, {order = #NHWC}>
    %1 = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %0, %1 : tensor<1x16x720x1280xf16, {order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       [[MAX_POOL0:%.+]] = VPU.NCE.MaxPool([[ARG0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>
    // CHECK:       [[MAX_POOL1:%.+]] = VPU.NCE.MaxPool([[ARG1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>
    // CHECK:       return [[MAX_POOL0]], [[MAX_POOL1]] : tensor<1x16x720x1280xf16, {order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteMaxPoolSameStrategy
func.func @PermuteMaxPoolSameStrategy(%arg0: tensor<1x16x720x1280xf16, {order = #NCHW}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>)
     -> (tensor<1x16x720x1280xf16, {order = #NCHW}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x16x720x1280xf16, {order = #NCHW}>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x1280xf16, {order = #NHWC}>
    %1 = VPU.NCE.MaxPool(%0) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x720x1280xf16, {order = #NHWC}> -> tensor<1x16x720x1280xf16, {order = #NCHW}>
    // CHECK:       [[PERMUTE0:%.+]] = VPU.NCE.Permute([[ARG0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY_PERMUTE:[^>]+]]>
    // CHECK:       [[MAX_POOL0:%.+]] = VPU.NCE.MaxPool([[PERMUTE0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY_MAX_POOL:[^>]+]]>

    %2 = VPU.NCE.Permute(%arg1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.MaxPool(%2) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [1, 1]} : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:      [[PERMUTE1:%.+]] = VPU.NCE.Permute([[ARG1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY_PERMUTE]]>
    // CHECK:      [[MAX_POOL1:%.+]] = VPU.NCE.MaxPool([[PERMUTE1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY_MAX_POOL]]>

    return %1, %3 : tensor<1x16x720x1280xf16, {order = #NCHW}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       return [[MAX_POOL0]], [[MAX_POOL1]] : tensor<1x16x720x1280xf16, {order = #NCHW}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NCHW}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!inputBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputStaticType = tensor<1x16x1600x2560xf16, {order = #NHWC}>

!outputBoundedType = tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>
!outputStaticType = tensor<1x16x3200x5120xf16, {order = #NHWC}>

!dynamicSparseType = !VPU.SparseTensor<data=!inputBoundedType,
                                       sparsity_map=tensor<1x16x3201x5121xi1, {order = #NHWC}>,
                                       storage_element_table=tensor<1x1x3201x5121xi32, {order = #NHWC}>,
                                       #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

!staticSparseType = !VPU.SparseTensor<data=!inputStaticType,
                                       sparsity_map=tensor<1x16x3201x5121xi1, {order = #NHWC}>,
                                       storage_element_table=tensor<1x1x3201x5121xi32, {order = #NHWC}>,
                                       #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>

func.func @SEPConvSameStrategy(%arg0: !inputBoundedType, %arg1: !inputStaticType)
    -> (!outputBoundedType, !outputStaticType) {

  %sparsity_map = const.Declare tensor<1x16x3201x5121xi1, {order = #NHWC}> = dense<1> : tensor<1x16x3201x5121xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x2x2xf16, {order = #NHWC}>

  %0 = VPU.StorageElementTable {
    dataElemType = f16, dataShape = [1, 16, 1600, 2560],
    seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [16]
  } -> tensor<1x1x3201x5121xi32, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

  %1 = VPU.GroupSparseTensor(%arg0, %sparsity_map, %0) {
    seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !dynamicSparseType

  %2 = VPU.NCE.Convolution(%1, %weights) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [16, 16, 2, 2], strides = [1, 1]}
    : !dynamicSparseType, tensor<16x16x2x2xf16, {order = #NHWC}> -> !outputBoundedType

  // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution
  // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>

  %3 = VPU.GroupSparseTensor(%arg1, %sparsity_map, %0) {
    seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !staticSparseType

  %4 = VPU.NCE.Convolution(%3, %weights) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [16, 16, 2, 2], strides = [1, 1]}
    : !staticSparseType, tensor<16x16x2x2xf16, {order = #NHWC}> -> !outputStaticType

  // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution
  // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>

  return %2, %4 : !outputBoundedType, !outputStaticType
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!inputDynamicType = tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>
!inputStaticType = tensor<1x12x1600x2560xf16, {order = #NHWC}>

!outputDynamicType = tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 3200, 5120]> : tensor<4xsi64>, order = #NHWC}>
!outputStaticType = tensor<1x3x3200x5120xf16, {order = #NHWC}>

// CHECK-LABEL: @D2SOpSameStrategy
func.func @D2SOpSameStrategy(%arg0: !inputStaticType, %arg1: !inputDynamicType) -> (!outputStaticType, !outputDynamicType) {
    // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x12x1600x2560xf16, {order = #NHWC}>
    // CHECK-SAME:  [[ARG1:%.+]]: tensor<1x12x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1600, 2560]> : tensor<4xsi64>, order = #NHWC}>

    %0 = VPU.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : !inputStaticType -> !outputStaticType

    // CHECK:       [[D2S_STATIC:%.+]] = VPU.DepthToSpace([[ARG0]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY:[^>]+]]>

    %1 = VPU.DepthToSpace(%arg1) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>}
        : !inputDynamicType -> !outputDynamicType

    // CHECK:      [[D2S_DYN:%.+]] = VPU.DepthToSpace([[ARG1]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<[[CLUSTER_STRATEGY]]>

    return %0, %1 : !outputStaticType, !outputDynamicType
    // CHECK:       return [[D2S_STATIC]], [[D2S_DYN]]
}
