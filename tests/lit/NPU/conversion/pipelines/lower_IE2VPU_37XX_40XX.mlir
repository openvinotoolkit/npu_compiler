//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --lower-IE-to-VPU %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

// CHECK: func.func @SingleLayer([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
func.func @SingleLayer(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    return %0 : tensor<1x1000xf16>

    // CHECK:  [[VAR0:%.+]] = VPU.SoftMax([[ARG0]]) {axisInd = 1 : i64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
    // CHECK:  return [[VAR0]] : tensor<1x1000xf16>
}

// -----

// CHECK: func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
func.func @ConstantLayer() -> tensor<1x2x2x2xf16> {
    %0 = const.Declare tensor<1x2x2x2xf16> = dense<1.0> : tensor<1x2x2x2xf16>
    return %0 : tensor<1x2x2x2xf16>

    // CHECK-DAG:  [[CST:%.+]] = const.Declare tensor<1x2x2x2xf16> = dense<1.000000e+00> : tensor<1x2x2x2xf16>
    // CHECK:  return [[CST]] : tensor<1x2x2x2xf16>
}

// -----

// CHECK: func.func @Reshape([[ARG0:%.+]]: tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
func.func @Reshape(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    return %0 : tensor<1x512xf32>

    // CHECK:  [[AFFINERESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    // CHECK:  return [[AFFINERESHAPE]] : tensor<1x512xf32>
}

// -----

// CHECK: func.func @ReshapeInGraph([[ARG0:%.+]]: tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32> {
func.func @ReshapeInGraph(%arg0 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32> {
    %0 = IE.Reshape(%arg0) { shape_value = [1, 512] } : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    %1 = IE.SoftMax(%0) {axisInd = 1} : tensor<1x512xf32> -> tensor<1x512xf32>
    %2 = IE.Reshape(%1) { shape_value = [1, 512, 1, 1] } : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    return %2 : tensor<1x512x1x1xf32>

    // CHECK:  [[AFFINERESHAPE0:%.+]] = VPU.AffineReshape([[ARG0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 512]} : tensor<1x512x1x1xf32> -> tensor<1x512xf32>
    // CHECK:  [[SOFTMAX:%.+]] = VPU.SoftMax([[AFFINERESHAPE0]]) {axisInd = 1 : i64} : tensor<1x512xf32> -> tensor<1x512xf32>
    // CHECK:  [[AFFINERESHAPE1:%.+]] = VPU.AffineReshape([[SOFTMAX]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2, 3]], shape_value = [1, 512, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1xf32>
    // CHECK:  return [[AFFINERESHAPE1]] : tensor<1x512x1x1xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @ConvToNCE([[ARG0:%.+]]: tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
func.func @ConvToNCE(%arg0: tensor<1x32x16x16xf16, {order = #NHWC}>) -> tensor<1x64x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x32x16x16xf16, {order = #NHWC}>, tensor<64x32x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16>
            -> tensor<1x64x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x64x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<64x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32>
    // CHECK:       [[OUT:%.+]] = VPU.NCE.Convolution([[ARG0]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) rawFilterShape [64, 32, 1, 1] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x64x16x16xf16, {order = #NHWC}>
    // CHECK:       return [[OUT]] : tensor<1x64x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DynamicExpand([[ARG0:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x20x20xf16>
func.func @DynamicExpand(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x20x20xf16> {
    %0 = IE.DynamicExpand(%arg0) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x20x20xf16>
    return %0 : tensor<1x3x20x20xf16>
    // CHECK-NOT:   IE.DynamicExpand
    // CHECK:       [[DynamicExpand:%.+]] = VPU.DynamicExpand([[ARG0]]) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x20x20xf16>
    // CHECK:       return [[DynamicExpand]] : tensor<1x3x20x20xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DynamicExpandU8([[ARG0:%.+]]: tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x20x20xui8>
func.func @DynamicExpandU8(%arg0: tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x3x20x20xui8> {
    %0 = IE.DynamicExpand(%arg0) : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x20x20xui8>
    return %0 : tensor<1x3x20x20xui8>
    // CHECK-NOT:   IE.DynamicExpand
    // CHECK:       [[DynamicExpand:%.+]] = VPU.DynamicExpand([[ARG0]]) : tensor<1x3x?x?xui8, {bounds = #const.OpaqueI64Elements<[1, 3, 20, 20]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x20x20xui8>
    // CHECK:       return [[DynamicExpand]] : tensor<1x3x20x20xui8>
}

// -----

// CHECK: func.func @ReinterpretCast([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<2x1000xi8> {
func.func @ReinterpretCast(%arg0: tensor<1x1000xf16>) -> tensor<2x1000xi8> {
    %0 = Core.ReinterpretCast(%arg0) : tensor<1x1000xf16> -> tensor<2x1000xi8>
    return %0 : tensor<2x1000xi8>

    // CHECK:  [[VAR0:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x1000xf16> -> tensor<2x1000xi8>
    // CHECK:  return [[VAR0]]
}
