//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --swap-operations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithBias
func.func @SwapWithBias(%arg0: tensor<4x9728x1x1xf16>) -> tensor<1x512x4x1xf16> {
   %filter = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>
    %3 = IE.Add(%2, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x4x1xf16>

    return %3 : tensor<1x512x4x1xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   // CHECK: IE.Transpose
   // CHECK-SAME: tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithBiasNHWC
func.func @SwapWithBiasNHWC(%arg0: tensor<4x9728x1x1xf16>) -> tensor<1x512x1x4xf16> {
   %filter = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %bias = const.Declare tensor<1x512x1x1xf16> = dense<1.000000e+00> : tensor<1x512x1x1xf16>

    %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x4x512x1xf16> -> tensor<1x512x1x4xf16>
    %3 = IE.Add(%2, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x1x4xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x4xf16>

    return %3 : tensor<1x512x1x4xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   // CHECK: IE.Transpose
   // CHECK-SAME: tensor<1x4x512x1xf16> -> tensor<1x512x1x4xf16>
}

// -----

// CHECK-LABEL: @NotChangeSwapBiasBroadcastWithReshape
func.func @NotChangeSwapBiasBroadcastWithReshape(%arg0: tensor<1x924x77x1xf16>) -> tensor<1x12x77x77xf16> {
   %bias = const.Declare tensor<1x1x77x77xf16> = dense<1.000000e+00> : tensor<1x1x77x77xf16>

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 12, 77, 77]} : tensor<1x924x77x1xf16> -> tensor<1x12x77x77xf16>
    %1 = IE.Add(%0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x77x77xf16>, tensor<1x1x77x77xf16> -> tensor<1x12x77x77xf16>

    return %1 : tensor<1x12x77x77xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x924x77x1xf16> -> tensor<1x12x77x77xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x12x77x77xf16>, tensor<1x1x77x77xf16> -> tensor<1x12x77x77xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotChangeSwapBiasWithReorder
func.func @NotChangeSwapBiasWithReorder(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %bias = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Reorder(%arg0) {dstOrder = #NHWC} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = IE.Add(%0, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

   // CHECK: IE.Reorder
   // CHECK-SAME: tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16, {order = #NHWC}>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithSingleValueBias
func.func @SwapWithSingleValueBias(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x512x4x1xf16> {
   %cst = const.Declare tensor<1x1x1x1xf16> = dense<-9.21613597> : tensor<1x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
   %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>
   %3 = IE.Add(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x4x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x4x1xf16>

   return %3 : tensor<1x512x4x1xf16>

   // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-9.21613597> : tensor<1x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
   // CHECK: [[ADD:%.*]] = IE.Add(%arg0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x512x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: [[RESHAPE:%.*]] = IE.AffineReshape([[ADD]])
   // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 512, 1]} : tensor<4x512x1x1xf16> -> tensor<1x4x512x1xf16>
   // CHECK: [[TRANS:%.*]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<1x4x512x1xf16> -> tensor<1x512x4x1xf16>
   // CHECK: return [[TRANS]] : tensor<1x512x4x1xf16>
}

// -----

// CHECK-LABEL: @SwapWithSingleValueBiasThroughConcat
func.func @SwapWithSingleValueBiasThroughConcat(%arg0: tensor<4096x4096x1x1xf16>, %arg1: tensor<4096x4096x1x1xf16>) -> tensor<1x2x4096x4096xf16> {
   %cst = const.Declare tensor<1x1x1x1xf16> = dense<-9.21613597> : tensor<1x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [4096, 4096]} : tensor<4096x4096x1x1xf16> -> tensor<4096x4096xf16>
   %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [4096, 4096]} : tensor<4096x4096x1x1xf16> -> tensor<4096x4096xf16>
   %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0], [4096, 0]]} : tensor<4096x4096xf16>, tensor<4096x4096xf16> -> tensor<8192x4096xf16>
   %3 = IE.AffineReshape(%2) {dim_mapping = [[0, 1], [2]], shape_value = [2, 4096, 4096]} : tensor<8192x4096xf16> -> tensor<2x4096x4096xf16>
   %4 = IE.Reshape(%3) {shape_value = [1, 2, 4096, 4096]} : tensor<2x4096x4096xf16> -> tensor<1x2x4096x4096xf16>
   %5 = IE.Add(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x4096x4096xf16>, tensor<1x1x1x1xf16> -> tensor<1x2x4096x4096xf16>

   return %5 : tensor<1x2x4096x4096xf16>

   // CHECK: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-9.21613597> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
   // CHECK: [[ADD0:%.*]] = IE.Add(%arg0, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<4096x4096x1x1xf16>
   // CHECK: [[RESHAPE0:%.*]] = IE.AffineReshape([[ADD0]])
   // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [4096, 4096]} : tensor<4096x4096x1x1xf16> -> tensor<4096x4096xf16>
   // CHECK: [[ADD1:%.*]] = IE.Add(%arg1, [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4096x4096x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<4096x4096x1x1xf16>
   // CHECK: [[RESHAPE1:%.*]] = IE.AffineReshape([[ADD1]])
   // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [4096, 4096]} : tensor<4096x4096x1x1xf16> -> tensor<4096x4096xf16>
   // CHECK: [[CONCAT:%.*]] = IE.Concat([[RESHAPE0]], [[RESHAPE1]])
   // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0], [4096, 0]]} : tensor<4096x4096xf16>, tensor<4096x4096xf16> -> tensor<8192x4096xf16>
   // CHECK: [[RESHAPE2:%.*]] = IE.AffineReshape([[CONCAT]])
   // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1], [2]], shape_value = [2, 4096, 4096]} : tensor<8192x4096xf16> -> tensor<2x4096x4096xf16>
   // CHECK: [[RESHAPE3:%.*]] = IE.Reshape([[RESHAPE2]]) {shape_value = [1, 2, 4096, 4096]} : tensor<2x4096x4096xf16> -> tensor<1x2x4096x4096xf16>
   // CHECK: return [[RESHAPE3]] : tensor<1x2x4096x4096xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithRelu
func.func @SwapWithRelu(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.ReLU(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.ReLU
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithSigmoid
func.func @SwapWithSigmoid(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Sigmoid(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Sigmoid
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithTanh
func.func @SwapWithTanh(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Tanh(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Tanh
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
}

// -----

// CHECK-LABEL: @SwapShapeCastWithTanh
func.func @SwapShapeCastWithTanh(%arg0: tensor<1x48x288x32xf16>) -> tensor<1x3x288x512xf16> {
    %shapecast = IE.ShapeCast {shape = [1, 3, 288, 512]} inputs(%arg0 : tensor<1x48x288x32xf16>) -> tensor<1x3x288x512xf16>
    %tanh = IE.Tanh(%shapecast) : tensor<1x3x288x512xf16> -> tensor<1x3x288x512xf16>
    return %tanh : tensor<1x3x288x512xf16>

    // CHECK: IE.Tanh
    // CHECK-SAME: -> tensor<1x48x288x32xf16>
    // CHECK: IE.ShapeCast
    // CHECK-SAME: -> tensor<1x3x288x512xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithClamp
func.func @SwapWithClamp(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
   %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
   %3 = IE.Clamp(%2) {min = 1.0, max = 3.0} : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Clamp
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithGelu
func.func @SwapWithGelu(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Gelu(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Gelu
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.AffineReshape
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithTwoClamps
func.func @SwapWithTwoClamps(%arg0: tensor<4x512x1x1xf16>) -> tensor<4x1x2048x1xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %1 = IE.Clamp(%0) {min = 1.0, max = 13.0} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
   %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<4x2048x1x1xf16> -> tensor<4x1x2048x1xf16>
   %3 = IE.Clamp(%2) {min = 4.0, max = 9.0} : tensor<4x1x2048x1xf16> -> tensor<4x1x2048x1xf16>

    return %3 : tensor<4x1x2048x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Clamp
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Clamp
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Transpose
    // CHECK-SAME: tensor<4x2048x1x1xf16> -> tensor<4x1x2048x1xf16>
}

// -----

// CHECK-LABEL: @SwapReshapeWithBiasHasSingleNonTrivialDim
func.func @SwapReshapeWithBiasHasSingleNonTrivialDim(%arg0: tensor<1x512x1x1xf16>) -> tensor<1x1x1x512xf16> {
   %filter = const.Declare tensor<512x512x1x1xf16> = dense<1.000000e+00> : tensor<512x512xf32>, [#const.Reshape<[512, 512, 1, 1]>, #const.CastElemType<f16>]
   %cst = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x512xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 512]>]


   %0 = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x1xf16>, tensor<512x512x1x1xf16> -> tensor<1x512x1x1xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>
   %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>

   return %2 : tensor<1x1x1x512xf16>

   // CHECK: IE.Convolution(%arg0, %cst_0)
   // CHECK-SAME: tensor<1x512x1x1xf16>, tensor<512x512x1x1xf16> -> tensor<1x512x1x1xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x512x1x1xf16>, tensor<1x512x1x1xf16> -> tensor<1x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL: @NoSwapReshapeWithEltwise
func.func @NoSwapReshapeWithEltwise(%arg0: tensor<4x9728x1x1xf16>, %arg1: tensor<4x9728x1x1xf16>) -> tensor<1x4x1x512xf16> {
   %filter_0 = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>
   %filter_1 = const.Declare tensor<512x9728x1x1xf16> = dense<1.000000e+00> : tensor<512x9728x1x1xf16>

   %0 = IE.Convolution(%arg0, %filter_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   %1 = IE.Convolution(%arg1, %filter_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   %2 = IE.AffineReshape(%0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4, 1, 512]} : tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   %3 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 4, 1, 512]} : tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   %4 = IE.Add(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x4x1x512xf16>, tensor<1x4x1x512xf16> -> tensor<1x4x1x512xf16>

   return %4 : tensor<1x4x1x512xf16>

   // CHECK: IE.Convolution(%arg0, %cst)
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.Convolution(%arg1, %cst)
   // CHECK-SAME: tensor<4x9728x1x1xf16>, tensor<512x9728x1x1xf16> -> tensor<4x512x1x1xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<4x512x1x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x4x1x512xf16>, tensor<1x4x1x512xf16> -> tensor<1x4x1x512xf16>
}

// -----

// CHECK-LABEL: @NoSwapReshapeWithLess4DBias
func.func @NoSwapReshapeWithLess4DBias(%arg0: tensor<4x16x1xf16>) -> tensor<1x4x1x512xf16> {
   %filter_0 = const.Declare tensor<512x16x1xf16> = dense<1.000000e+00> : tensor<512x16x1xf16>
   %bias = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x1x512xf16>

   %0 = IE.Convolution(%arg0, %filter_0) {dilations = [1], pads_begin = [0], pads_end = [0], strides = [1]} : tensor<4x16x1xf16>, tensor<512x16x1xf16> -> tensor<4x512x1xf16>
   %1 = IE.Reshape(%0) {shape_value = [1, 4, 1, 512]} : tensor<4x512x1xf16> -> tensor<1x4x1x512xf16>
   %2 = IE.Add(%1, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x4x1x512xf16>

   return %2 : tensor<1x4x1x512xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<4x16x1xf16>, tensor<512x16x1xf16> -> tensor<4x512x1xf16>
   // CHECK: IE.Reshape
   // CHECK-SAME: tensor<4x512x1xf16> -> tensor<1x4x1x512xf16>
   // CHECK: IE.Add
   // CHECK-SAME: tensor<1x4x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x4x1x512xf16>
}

// -----

// CHECK-LABEL: @NoSwapReshapeWithMoreThan4DSigmoid
func.func @NoSwapReshapeWithMoreThan4DSigmoid(%arg0: tensor<1x3x9x16x1xf16> ) -> tensor<3x9x16x1xf16> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 9, 16, 1]} : tensor<1x3x9x16x1xf16> -> tensor<3x9x16x1xf16>
   %1 = IE.Sigmoid(%0) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>

   return %1 : tensor<3x9x16x1xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x3x9x16x1xf16> -> tensor<3x9x16x1xf16>
   // CHECK: IE.Sigmoid
   // CHECK-SAME: tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
}

// -----

// CHECK-LABEL: @OptimizeSliceTanH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @OptimizeSliceTanH(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x8x32x32xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x8x32x32xf16>
   %1 = IE.Tanh(%0) : tensor<1x8x32x32xf16> -> tensor<1x8x32x32xf16>
   return %1 : tensor<1x8x32x32xf16>

   // CHECK:        [[TANH:%.+]] = IE.Tanh([[INPUT]]) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   // CHECK:        [[SLICE:%.+]] = IE.Slice [[TANH]] [0, 0, 0, 0] [1, 8, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x8x32x32xf16>
}

// -----

// CHECK-LABEL: @DoNotOptimizeSliceTanH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @DoNotOptimizeSliceTanH(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x4x32x32xf16> {
   %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   %1 = IE.Tanh(%0) : tensor<1x4x32x32xf16> -> tensor<1x4x32x32xf16>
   return %1 : tensor<1x4x32x32xf16>

   // CHECK:        [[SLICE:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 32, 32] : tensor<1x16x32x32xf16> to tensor<1x4x32x32xf16>
   // CHECK:        [[TANH:%.+]] = IE.Tanh([[SLICE]]) : tensor<1x4x32x32xf16> -> tensor<1x4x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @OptimizeSigmoidReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16, {order = #NHWC}>
func.func @OptimizeSigmoidReorder(%arg0: tensor<1x16x32x32xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
   %1 = IE.Sigmoid(%0) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %1 : tensor<1x16x32x32xf16>

   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[INPUT]]) : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[SIGMOID]]) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SwapWithBiasOrderChanged
func.func @SwapWithBiasOrderChanged(%arg0: tensor<8x64x1x1xf16, {order = #NHWC}>) -> tensor<1x8x64x1xf16, {order = #NCWH}> {
   %bias = const.Declare tensor<1x1x64x1xf16, {order = #NCWH}> = dense<1.000000e+00> : tensor<1x1x64x1xf16>, [#const.Reorder<#NCWH>]

    %1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 8, 64, 1]} : tensor<8x64x1x1xf16, {order = #NHWC}> -> tensor<1x8x64x1xf16, {order = #NCWH}>
    %2 = IE.Add(%1, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8x64x1xf16, {order = #NCWH}>, tensor<1x1x64x1xf16, {order = #NCWH}> -> tensor<1x8x64x1xf16, {order = #NCWH}>

    return %2 : tensor<1x8x64x1xf16, {order = #NCWH}>

   // CHECK: const.Declare
   // CHECK-SAME:  #const.Reshape<[1, 64, 1, 1]>, #const.Reorder<#NHWC>]
   // CHECK: IE.Add
   // CHECK-SAME: tensor<8x64x1x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<8x64x1x1xf16, {order = #NHWC}>
   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<8x64x1x1xf16, {order = #NHWC}> -> tensor<1x8x64x1xf16, {order = #NCWH}>
}

// -----

// CHECK-LABEL: @SwapConcatWithClamp
func.func @SwapConcatWithClamp(%arg0: tensor<4x512x1x1xf16>, %arg1: tensor<4x512x1x1xf16>) -> tensor<4x2048x1x2xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %cst_1 = const.Declare tensor<2048x512x1x1xf16> = dense<2.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %1 = IE.Convolution(%arg1, %cst_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<4x2048x1x1xf16>, tensor<4x2048x1x1xf16> -> tensor<4x2048x1x2xf16>
   %3 = IE.Clamp(%2) {min = 1.0, max = 3.0} : tensor<4x2048x1x2xf16> -> tensor<4x2048x1x2xf16>

   return %3 : tensor<4x2048x1x2xf16>

   // CHECK:      [[FILTER_1:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   // CHECK:      [[FILTER_2:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<2.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   // CHECK:      [[CONV_1:%.*]] = IE.Convolution(%arg0, [[FILTER_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   // CHECK:      [[CONV_2:%.*]] = IE.Convolution(%arg1, [[FILTER_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   // CHECK:      [[CLAMP_1:%.*]] = IE.Clamp([[CONV_1]]) {max = 3.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
   // CHECK:      [[CLAMP_2:%.*]] = IE.Clamp([[CONV_2]]) {max = 3.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
   // CHECK:      [[CONCAT:%.*]] = IE.Concat([[CLAMP_1]], [[CLAMP_2]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<4x2048x1x1xf16>, tensor<4x2048x1x1xf16> -> tensor<4x2048x1x2xf16>
   // CHECK:      return [[CONCAT]] : tensor<4x2048x1x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.0>
!qElemType1 = !quant.uniform<u8:f16, 2.0>
// CHECK-LABEL: @SwapExpandWithQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x416x416x!qElemType>
func.func @SwapExpandWithQuantizeCast(%arg0: tensor<1x3x416x416x!qElemType>) -> tensor<1x4x416x416x!qElemType1> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x416x416x!qElemType>
            -> tensor<1x4x416x416x!qElemType>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x4x416x416x!qElemType>
            -> tensor<1x4x416x416x!qElemType1>
    return %1 : tensor<1x4x416x416x!qElemType1>

    // CHECK:       [[QUANTCAST:%.*]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType1} : tensor<1x3x416x416x!qElemType>
    // CHECK-SAME:          -> tensor<1x3x416x416x!qElemType1>
    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[QUANTCAST]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x416x416x!qElemType1>
    // CHECK-SAME:          -> tensor<1x4x416x416x!qElemType1>
    // CHECK:       return [[EXPAND]] : tensor<1x4x416x416x!qElemType1>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0:124, 1.0:124, 1.0:124}>
!qElemType1 = !quant.uniform<u8:f16:1, {2.0:124, 2.0:124, 2.0:124, 2.0:124}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.0:124, 1.0:124, 1.0:124, 1.0:124}>
// CHECK-LABEL: @SkipSwapExpandWithPerChannelQuantizeCast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x416x416x!qElemType>
func.func @SkipSwapExpandWithPerChannelQuantizeCast(%arg0: tensor<1x3x416x416x!qElemType>) -> tensor<1x4x416x416x!qElemType1> {
    %0 = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x416x416x!qElemType>
            -> tensor<1x4x416x416x!qElemType2>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x4x416x416x!qElemType2>
            -> tensor<1x4x416x416x!qElemType1>
    return %1 : tensor<1x4x416x416x!qElemType1>

    // CHECK:       [[EXPAND:%.*]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} : tensor<1x3x416x416x!qElemType>
    // CHECK-SAME:          -> tensor<1x4x416x416x!qElemType2>
    // CHECK:       [[QUANTCAST:%.*]] = IE.QuantizeCast([[EXPAND]]) {dstElemType = !qElemType1} : tensor<1x4x416x416x!qElemType2>
    // CHECK-SAME:          -> tensor<1x4x416x416x!qElemType1>
    // CHECK:       return [[QUANTCAST]] : tensor<1x4x416x416x!qElemType1>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithClampAndLRelu
func.func @SwapWithClampAndLRelu(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
   %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
   %3 = IE.Clamp(%2) {min = 1.0, max = 2.0} : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>
   %4 = IE.LeakyRelu(%3) {negative_slope = 2.500000e-01 : f64} : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %4 : tensor<1x2048x4x1xf16>

  // CHECK: [[FILTER_1:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
  //CHECK: [[VAL0:%.*]] =  IE.Convolution(%arg0, [[FILTER_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
  //CHECK: [[VAL1:%.*]] =  IE.LeakyRelu([[VAL0]]) {negative_slope = 2.500000e-01 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
  //CHECK: [[VAL2:%.*]] =  IE.Clamp([[VAL1]]) {max = 2.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
  //CHECK: [[VAL3:%.*]] =  IE.AffineReshape([[VAL2]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
  //CHECK: [[VAL4:%.*]] =  IE.Transpose([[VAL3]]) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
  //CHECK: return [[VAL4]] :  tensor<1x2048x4x1xf16>
}

// -----

// CHECK-LABEL: @SwapWithAffineReshapeAndLRelu
func.func @SwapWithAffineReshapeAndLRelu(%arg0: tensor<4x2048x1x1xf16>) -> tensor<1x4x2048x1xf16> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
   %1 = IE.LeakyRelu(%0) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1xf16> -> tensor<1x4x2048x1xf16>

    return %1 : tensor<1x4x2048x1xf16>

  //CHECK: [[VAL0:%.*]] =   IE.LeakyRelu(%arg0) {negative_slope = 2.500000e-01 : f64} : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
  //CHECK: [[VAL1:%.*]] =  IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>

  //CHECK: return [[VAL1]] :  tensor<1x4x2048x1xf16>
}

// -----

// CHECK-LABEL: @NoSwapWithAffineReshapeAndLReluNot4D
func.func @NoSwapWithAffineReshapeAndLReluNot4D(%arg0: tensor<3x62x62xf16>) -> tensor<1x3x62x62xf16> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 3, 62, 62]} : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>
   %1 = IE.LeakyRelu(%0) {negative_slope = 2.500000e-01 : f64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>

    return %1 : tensor<1x3x62x62xf16>

  //CHECK: [[VAL0:%.*]] =  IE.AffineReshape(%arg0)
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 3, 62, 62]} : tensor<3x62x62xf16> -> tensor<1x3x62x62xf16>
  //CHECK: [[VAL1:%.*]] = IE.LeakyRelu([[VAL0]]) {negative_slope = 2.500000e-01 : f64} : tensor<1x3x62x62xf16> -> tensor<1x3x62x62xf16>

  //CHECK: return [[VAL1]] :  tensor<1x3x62x62xf16>

}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @NoSwapWithActivation
func.func @NoSwapWithActivation(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x4x2048x1xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
   %2 = IE.MaxPool(%1) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x2048x1xf16> -> tensor<1x4x2048x1xf16>
   %3 = IE.LeakyRelu(%2) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1xf16> -> tensor<1x4x2048x1xf16>

    return %3 : tensor<1x4x2048x1xf16>

  // CHECK: [[FILTER_1:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
  //CHECK: [[VAL0:%.*]] = IE.Convolution(%arg0, [[FILTER_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
  //CHECK: [[VAL1:%.*]] =  IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
  //CHECK: [[VAL2:%.*]] =   IE.MaxPool([[VAL1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x4x2048x1xf16> -> tensor<1x4x2048x1xf16>
  //CHECK: [[VAL3:%.*]] = IE.LeakyRelu([[VAL2]]) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1xf16> -> tensor<1x4x2048x1xf16>

  //CHECK: return [[VAL3]] :  tensor<1x4x2048x1xf16>

}

// -----

// CHECK-LABEL: @SwapWithAffineReshapeAndExp
func.func @SwapWithAffineReshapeAndExp(%arg0: tensor<1x40x23x4xf16>) -> tensor<1x1x920x4xf16> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>
   %1 = IE.Exp(%0) : tensor<1x1x920x4xf16> -> tensor<1x1x920x4xf16>

   return %1 : tensor<1x1x920x4xf16>

  //CHECK: [[VAL0:%.*]] =  IE.Exp(%arg0) : tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>
  //CHECK: [[VAL1:%.*]] =  IE.AffineReshape([[VAL0]])
  //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>

  //CHECK: return [[VAL1]] :  tensor<1x1x920x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithTransposeAndExp
func.func @SwapWithTransposeAndExp(%arg0: tensor<1x4x40x23xf16>) -> tensor<1x40x23x4xf16> {
   %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>
   %1 = IE.Exp(%0) :  tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>

   return %1 : tensor<1x40x23x4xf16>

  //CHECK: [[VAL0:%.*]] =  IE.Exp(%arg0) : tensor<1x4x40x23xf16> -> tensor<1x4x40x23xf16>
  //CHECK: [[VAL1:%.*]] =  IE.Transpose([[VAL0]])
  //CHECK-SAME{LITERAL}: {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>

  //CHECK: return [[VAL1]] :  tensor<1x40x23x4xf16>
}

// -----

// CHECK-LABEL: @SwapWithBiasCostDefinedAfterActivation
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<768x1152x3xf16>)
func.func @SwapWithBiasCostDefinedAfterActivation(%arg0: tensor<768x1152x3xf16>) -> tensor<1x3x768x1152xf16> {
  %fq_out_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.221250e+02> : tensor<1x1x1x1xf16>
  %fq_in_out_high = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
  %fq_in_low = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
  %bias = const.Declare tensor<1x3x1x1xf16> = dense<[[[223]], [[15]], [[181]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.Rescale<-1.000000e+00 : f64>]
  %affine_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 768, 1152, 3]} : tensor<768x1152x3xf16> -> tensor<1x768x1152x3xf16>
  %transpose = IE.Transpose(%affine_reshape) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x768x1152x3xf16> -> tensor<1x3x768x1152xf16>
  %fq = IE.FakeQuantize(%bias, %fq_in_low, %fq_in_out_high, %fq_out_low, %fq_in_out_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
  %add = IE.Add(%transpose, %fq) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x768x1152xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x768x1152xf16>

  return %add : tensor<1x3x768x1152xf16>

   // CHECK: [[FQ_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.221250e+02> : tensor<1x1x1x1xf16>
   // CHECK: [[FQ_IN_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
   // CHECK: [[FQ_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.550000e+02> : tensor<1x1x1x1xf16>
   // CHECK: [[BIAS:%.+]] = const.Declare tensor<1x3x1x1xf16>

   // CHECK-SAME{LITERAL}: dense<[[[223]], [[15]], [[181]]]> : tensor<3x1x1xui8>, [#const.Reshape<[1, 3, 1, 1]>, #const.CastElemType<f32>, #const.CastElemType<f16>, #const.Rescale<-1.000000e+00 : f64>]
   // CHECK: [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
   // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 768, 1152, 3]} : tensor<768x1152x3xf16> -> tensor<1x768x1152x3xf16>
   // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[BIAS]], [[FQ_IN_LOW]], [[FQ_IN_OUT_HIGH]], [[FQ_OUT_LOW]], [[FQ_IN_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x1x1xf16>
   // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[FQ]]) {order_value = #NHWC} : tensor<1x3x1x1xf16> -> tensor<1x1x1x3xf16>
   // CHECK: [[ADD:%.+]] = IE.Add([[AFFINE_RESHAPE]], [[TRANSPOSE_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768x1152x3xf16>, tensor<1x1x1x3xf16> -> tensor<1x768x1152x3xf16>
   // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[ADD]]) {order_value = #NWCH} : tensor<1x768x1152x3xf16> -> tensor<1x3x768x1152xf16>

   // CHECK: return [[TRANSPOSE_1]] : tensor<1x3x768x1152xf16>
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.034064797794117647:55>
// CHECK-LABEL: @SwapWithAffineReshapeAndLReluQ
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<4x2048x1x1x!qElemType>)
func.func @SwapWithAffineReshapeAndLReluQ(%arg0: tensor<4x2048x1x1x!qElemType>) -> tensor<1x4x2048x1x!qElemType> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1x!qElemType> -> tensor<1x4x2048x1x!qElemType>
   %1 = IE.LeakyRelu(%0) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1x!qElemType> -> tensor<1x4x2048x1x!qElemType>
   return %1 : tensor<1x4x2048x1x!qElemType>

   //CHECK: [[LEAKY_RELU:%.+]] =   IE.LeakyRelu([[INPUT]]) {negative_slope = 2.500000e-01 : f64} : tensor<4x2048x1x1x!qElemType> -> tensor<4x2048x1x1x!qElemType>
   //CHECK: [[AFFINE_RESHAPE:%.+]] =  IE.AffineReshape([[LEAKY_RELU]])
   //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1x!qElemType> -> tensor<1x4x2048x1x!qElemType>
   //CHECK: return [[AFFINE_RESHAPE]] :  tensor<1x4x2048x1x!qElemType>

}

// -----
!qElemType = !quant.uniform<u8:f16:0, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {2.000000e-01:128,3.000000e-01:128,4.000000e-01:128,5.000000e-01:128}>
!qElemType2 = !quant.uniform<u8:f16:1, {1.000000e-01:128,2.000000e-01:128,3.000000e-01:128,4.000000e-01:128}>

// CHECK-LABEL: @DoNotSwapWithAffineReshapeAndLReluPerAxis
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<4x2048x1x1x!qElemType>)
func.func @DoNotSwapWithAffineReshapeAndLReluPerAxis(%arg0: tensor<4x2048x1x1x!qElemType>) -> tensor<1x4x2048x1x!qElemType1> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1x!qElemType> -> tensor<1x4x2048x1x!qElemType2>
   %1 = IE.LeakyRelu(%0) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1x!qElemType2> -> tensor<1x4x2048x1x!qElemType1>
   return %1 : tensor<1x4x2048x1x!qElemType1>

   //CHECK:               [[AFFINE_RESHAPE:%.+]] =   IE.AffineReshape([[INPUT]])
   //CHECK-SAME{LITERAL}:    {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]}
   //CHECK-SAME:             tensor<4x2048x1x1x!qElemType> -> tensor<1x4x2048x1x!qElemType2>
   //CHECK:                [[LEAKY_RELU:%.+]] =  IE.LeakyRelu([[AFFINE_RESHAPE]])
   //CHECK-SAME:              {negative_slope = 2.500000e-01 : f64}
   //CHECK-SAME:              tensor<1x4x2048x1x!qElemType2> -> tensor<1x4x2048x1x!qElemType1>
   //CHECK: return [[LEAKY_RELU]] :  tensor<1x4x2048x1x!qElemType1>

}

// -----
!qElemType = !quant.uniform<u8:f16, 1.0:124>
!qElemType1 = !quant.uniform<u8:f16, 2.0:124>
// CHECK-LABEL: @SwapWithAffineReshapeAndLReluDiffTypes
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<4x2048x1x1x!qElemType>)
func.func @SwapWithAffineReshapeAndLReluDiffTypes(%arg0: tensor<4x2048x1x1x!qElemType>) -> tensor<1x4x2048x1x!qElemType1> {
   %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1x!qElemType> -> tensor<1x4x2048x1x!qElemType>
   %1 = IE.LeakyRelu(%0) {negative_slope = 2.500000e-01 : f64} : tensor<1x4x2048x1x!qElemType> -> tensor<1x4x2048x1x!qElemType1>
   return %1 : tensor<1x4x2048x1x!qElemType1>

   //CHECK: [[LEAKY_RELU:%.+]] =   IE.LeakyRelu([[INPUT]]) {negative_slope = 2.500000e-01 : f64} : tensor<4x2048x1x1x!qElemType> -> tensor<4x2048x1x1x!qElemType1>
   //CHECK: [[AFFINE_RESHAPE:%.+]] =  IE.AffineReshape([[LEAKY_RELU]])
   //CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1x!qElemType1> -> tensor<1x4x2048x1x!qElemType1>
   //CHECK: return [[AFFINE_RESHAPE]] :  tensor<1x4x2048x1x!qElemType1>

}


// -----
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
!qElemType = !quant.uniform<u8:f16, 0.0045055291231940776:174>

// CHECK-LABEL: @OptimizeDequantizeMemPermuteReorder()
func.func @OptimizeDequantizeMemPermuteReorder() -> tensor<320x320x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
   %weights = const.Declare tensor<320x320x3x3x!qElemType> = dense<1> :
               tensor<320x320x3x3xui8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<ui8>,
               #const.CastElemType<!qElemType>]
   %dequantize = IE.Dequantize(%weights) {dstElemType = f16} : tensor<320x320x3x3x!qElemType> -> tensor<320x320x3x3xf16>
   %mem_permute = IE.MemPermute(%dequantize) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} :
                  tensor<320x320x3x3xf16> -> tensor<320x320x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
   return %mem_permute : tensor<320x320x3x3xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

   // CHECK:        [[CST:%.+]] = const.Declare tensor<320x320x3x3x!qElemType, {order = #NHWC}>
   // CHECK-SAME:   #const.MemPermute<#NHWC, #NHWC>
   // CHECK:        [[DEQUANT:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<320x320x3x3x!qElemType, {order = #NHWC}> -> tensor<320x320x3x3xf16, {order = #NHWC}>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithFQBias
func.func @SwapWithFQBias(%arg0: tensor<77x1024x1x1xf16>) -> tensor<1x1024x77x1xf16> {
   %filter = const.Declare tensor<1024x1024x1x1xf16> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>
   %bias = const.Declare tensor<1x1024x1x1xf16> = dense<1.000000e+00> : tensor<1x1024x1x1xf16>
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>

   %conv = IE.Convolution(%arg0, %filter) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<77x1024x1x1xf16>, tensor<1024x1024x1x1xf16> -> tensor<77x1024x1x1xf16>
   %reshape0 = IE.AffineReshape(%conv) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 77, 1024]} : tensor<77x1024x1x1xf16> -> tensor<1x1x77x1024xf16>
   %fq = IE.FakeQuantize(%reshape0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   %reshape1 = IE.AffineReshape(%fq) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 77, 1024, 1]} : tensor<1x1x77x1024xf16> -> tensor<1x77x1024x1xf16>
   %transpose = IE.Transpose(%reshape1) {order_value = #NHCW} : tensor<1x77x1024x1xf16> -> tensor<1x1024x77x1xf16>
   %add = IE.Add(%transpose, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x77x1xf16>, tensor<1x1024x1x1xf16> -> tensor<1x1024x77x1xf16>

   return %add : tensor<1x1024x77x1xf16>

   // CHECK: IE.Convolution
   // CHECK-SAME: tensor<77x1024x1x1xf16>, tensor<1024x1024x1x1xf16> -> tensor<77x1024x1x1xf16>
   // CHECK-NEXT: IE.FakeQuantize
   // CHECK-SAME: tensor<77x1024x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<77x1024x1x1xf16>
   // CHECK-NEXT: IE.Add
   // CHECK-SAME: tensor<77x1024x1x1xf16>, tensor<1x1024x1x1xf16> -> tensor<77x1024x1x1xf16>
   // CHECK-NEXT: IE.Reshape
   // CHECK-SAME: tensor<77x1024x1x1xf16> -> tensor<1x77x1024x1xf16>
   // CHECK-NEXT: IE.Transpose
   // CHECK-SAME: tensor<1x77x1024x1xf16> -> tensor<1x1024x77x1xf16>
}

// -----

// CHECK-LABEL: @DoNotSwapWithFQGelu
func.func @DoNotSwapWithFQGelu(%arg0: tensor<256x3072x1x1xf16>) -> tensor<1x1x256x3072xf16> {
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>

   %reshape1 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 256, 3072]} : tensor<256x3072x1x1xf16> -> tensor<1x1x256x3072xf16>
   %fq1 = IE.FakeQuantize(%reshape1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x256x3072xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x3072xf16>
   %gelu = IE.Gelu(%fq1) : tensor<1x1x256x3072xf16> -> tensor<1x1x256x3072xf16>
   %fq2 = IE.FakeQuantize(%gelu, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x256x3072xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x3072xf16>

   return %fq2 : tensor<1x1x256x3072xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<256x3072x1x1xf16> -> tensor<1x1x256x3072xf16>
   // CHECK-NEXT: IE.FakeQuantize
   // CHECK-SAME: tensor<1x1x256x3072xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x3072xf16>
   // CHECK-NEXT: IE.Gelu
   // CHECK-SAME: tensor<1x1x256x3072xf16> -> tensor<1x1x256x3072xf16>
   // CHECK-NEXT: IE.FakeQuantize
   // CHECK-SAME: tensor<1x1x256x3072xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x3072xf16>
}

// -----

// CHECK-LABEL: @DoNotSwapWithFQSingleValueBiasAddMul
func.func @DoNotSwapWithFQSingleValueBiasAddMul(%arg0: tensor<1x77x1024x1xf16>) -> tensor<1x1x77x1024xf16> {
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>
   %bias = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+02> : tensor<1x1x1x1xf16>
   %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.5> : tensor<1x1x1x1xf16>

   %reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 77, 1024]} : tensor<1x77x1024x1xf16> -> tensor<1x1x77x1024xf16>
   %fq = IE.FakeQuantize(%reshape, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   %add = IE.Add(%fq, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   %multiply = IE.Multiply(%add, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>

   return %multiply : tensor<1x1x77x1024xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<1x77x1024x1xf16> -> tensor<1x1x77x1024xf16>
   // CHECK-NEXT: IE.FakeQuantize
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   // CHECK-NEXT: IE.Add
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   // CHECK-NEXT: IE.Multiply
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
}

// -----

// CHECK-LABEL: @SwapWithFQNotConstAdd
func.func @SwapWithFQNotConstAdd(%arg0: tensor<1x77x1024x1xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x77x1024xf16> {
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>
   %scale = const.Declare tensor<1x1x1x1xf16> = dense<0.5> : tensor<1x1x1x1xf16>

   %reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 77, 1024]} : tensor<1x77x1024x1xf16> -> tensor<1x1x77x1024xf16>
   %fq = IE.FakeQuantize(%reshape, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   %add = IE.Add(%fq, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   %multiply = IE.Multiply(%add, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>

   return %multiply : tensor<1x1x77x1024xf16>

   // CHECK: IE.FakeQuantize
   // CHECK-SAME: tensor<1x77x1024x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x77x1024x1xf16>
   // CHECK-NEXT: IE.AffineReshape
   // CHECK-SAME: tensor<1x77x1024x1xf16> -> tensor<1x1x77x1024xf16>
   // CHECK-NEXT: IE.Add
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   // CHECK-NEXT: IE.Multiply
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SwapWithFQChangeLayout
func.func @SwapWithFQChangeLayout(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}> {
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>

   %reshape0 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x1x512x4096xf16, {order = #NCWH}>
   %fq = IE.FakeQuantize(%reshape0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x512x4096xf16, {order = #NCWH}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x512x4096xf16, {order = #NCWH}>
   %reshape1 = IE.AffineReshape(%fq) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [1, 512, 4096, 1]} : tensor<1x1x512x4096xf16, {order = #NCWH}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>

   return %reshape1 : tensor<1x512x4096x1xf16, {order = #NHWC}>

   // CHECK: IE.FakeQuantize
   // CHECK-SAME: tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x512x64x64xf16, {order = #NHWC}>
   // CHECK-NEXT: IE.AffineReshape
   // CHECK-SAME: tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @NotSwapWithPerAxisFQ
func.func @NotSwapWithPerAxisFQ(%arg0: tensor<3x30x30x1xf16>) -> tensor<1x3x30x30xf16> {
   %low = const.Declare tensor<1x3x1x1xf16> = dense<[[[[0.0]],[[0.0]], [[0.0]]]]>  : tensor<1x3x1x1xf16>
   %high = const.Declare tensor<1x3x1x1xf16> = dense<[[[[5.0]],[[6.0]], [[5.0]]]]>  : tensor<1x3x1x1xf16>

   %reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 3, 30, 30]} : tensor<3x30x30x1xf16> -> tensor<1x3x30x30xf16>
   %fq = IE.FakeQuantize(%reshape, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x3x30x30xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x30x30xf16>

   return %fq : tensor<1x3x30x30xf16>

   // CHECK: IE.AffineReshape
   // CHECK-SAME: tensor<3x30x30x1xf16> -> tensor<1x3x30x30xf16>
   // CHECK-NEXT: IE.FakeQuantize
   // CHECK-SAME: tensor<1x3x30x30xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x30x30xf16>
}

// -----

// CHECK-LABEL: @FQInputIsBlockArgument
func.func @FQInputIsBlockArgument(%arg0: tensor<1x1x77x1024xf16>) -> tensor<1x1x77x1024xf16> {
   %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.12660551> : tensor<1x1x1x1xf16>
   %high = const.Declare tensor<1x1x1x1xf16> = dense<5.930200e+00> : tensor<1x1x1x1xf16>
   %fq = IE.FakeQuantize(%arg0, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
   return %fq : tensor<1x1x77x1024xf16>

   // CHECK: IE.FakeQuantize
   // CHECK-SAME: tensor<1x1x77x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x77x1024xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @OptimizeSigmoidReorder
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16, {order = #NHCW}>
func.func @OptimizeSigmoidReorder(%arg0: tensor<1x16x32x32xf16, {order = #NHCW}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   %1 = IE.Sigmoid(%0) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %1 : tensor<1x16x32x32xf16>

   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[INPUT]]) : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16, {order = #NHCW}>
   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[SIGMOID]])  {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapBatchedMemPermuteSlice
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x16x126x224xf16>
func.func @SwapBatchedMemPermuteSlice(%arg0: tensor<4x16x126x224xf16>) -> (tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    %2 = IE.Slice %0 [1, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    %3 = IE.Slice %0 [2, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    %4 = IE.Slice %0 [3, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>

    return %1, %2, %3, %4 : tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM0:%.+]] = IE.MemPermute([[SLICE0]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [1, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM1:%.+]] = IE.MemPermute([[SLICE1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [2, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM2:%.+]] = IE.MemPermute([[SLICE2]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[INPUT]] [3, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM3:%.+]] = IE.MemPermute([[SLICE3]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapBatchedMemPermuteSliceWithPartialSliceCoverage
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x16x126x224xf16>
func.func @SwapBatchedMemPermuteSliceWithPartialSliceCoverage(%arg0: tensor<4x16x126x224xf16>) -> (tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    %2 = IE.Slice %0 [1, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<1x16x126x224xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM1:%.+]] = IE.MemPermute([[SLICE1]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [1, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16> to tensor<1x16x126x224xf16>
    // CHECK:       [[MEMPERM2:%.+]] = IE.MemPermute([[SLICE2]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x126x224xf16> -> tensor<1x16x126x224xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapBatchedMemPermuteSliceWithMultipleDimensions
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x16x126x224xf16>
func.func @NotSwapBatchedMemPermuteSliceWithMultipleDimensions(%arg0: tensor<4x16x126x224xf16>) -> tensor<1x8x63x112xf16, {order = #NHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 8, 63, 112] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x8x63x112xf16, {order = #NHWC}>

    return %1 : tensor<1x8x63x112xf16, {order = #NHWC}>

    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MEMPERM]] [0, 0, 0, 0] [1, 8, 63, 112] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x8x63x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapBatchedMemPermuteSliceWithNonSliceUser
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x16x126x224xf16>
func.func @NotSwapBatchedMemPermuteSliceWithNonSliceUser(%arg0: tensor<4x16x126x224xf16>) -> (tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<4x16x126x224xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>

    return %1, %0 : tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<4x16x126x224xf16, {order = #NHWC}>

    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MEMPERM]] [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotSwapBatchedMemPermuteSliceWithMismatchedBatchSize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4x16x126x224xf16>
func.func @NotSwapBatchedMemPermuteSliceWithMismatchedBatchSize(%arg0: tensor<4x16x126x224xf16>) -> (tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<2x16x126x224xf16, {order = #NHWC}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    %2 = IE.Slice %0 [1, 0, 0, 0] [2, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<2x16x126x224xf16, {order = #NHWC}>

    return %1, %2 : tensor<1x16x126x224xf16, {order = #NHWC}>, tensor<2x16x126x224xf16, {order = #NHWC}>

    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<4x16x126x224xf16> -> tensor<4x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[MEMPERM]] [0, 0, 0, 0] [1, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<1x16x126x224xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[MEMPERM]] [1, 0, 0, 0] [2, 16, 126, 224] : tensor<4x16x126x224xf16, {order = #NHWC}> to tensor<2x16x126x224xf16, {order = #NHWC}>
}

// -----

#HWCN = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1, d0)>

// CHECK-LABEL: @NotSwapBatchedMemPermuteSliceWithNDimPositionChange
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x32x2x16xf16>
func.func @NotSwapBatchedMemPermuteSliceWithNDimPositionChange(%arg0: tensor<2x32x2x16xf16>) -> (tensor<1x32x2x16xf16, {order = #HWCN}>, tensor<1x32x2x16xf16, {order = #HWCN}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #HWCN, mem_perm = #HWCN} : tensor<2x32x2x16xf16> -> tensor<2x32x2x16xf16, {order = #HWCN}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 32, 2, 16] : tensor<2x32x2x16xf16, {order = #HWCN}> to tensor<1x32x2x16xf16, {order = #HWCN}>
    %2 = IE.Slice %0 [1, 0, 0, 0] [1, 32, 2, 16] : tensor<2x32x2x16xf16, {order = #HWCN}> to tensor<1x32x2x16xf16, {order = #HWCN}>

    return %1, %2 : tensor<1x32x2x16xf16, {order = #HWCN}>, tensor<1x32x2x16xf16, {order = #HWCN}>

    // CHECK:       [[MEMPERM:%.+]] = IE.MemPermute([[INPUT]]) {dst_order = #map, mem_perm = #map} : tensor<2x32x2x16xf16> -> tensor<2x32x2x16xf16, {order = #map}>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[MEMPERM]] [0, 0, 0, 0] [1, 32, 2, 16] : tensor<2x32x2x16xf16, {order = #map}> to tensor<1x32x2x16xf16, {order = #map}>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[MEMPERM]] [1, 0, 0, 0] [1, 32, 2, 16] : tensor<2x32x2x16xf16, {order = #map}> to tensor<1x32x2x16xf16, {order = #map}>
    // CHECK:       return [[SLICE1]], [[SLICE2]]
}

// -----

#HWCN = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1, d0)>
#CNHW = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#CNWH = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>

// CHECK-LABEL: @SwapBatchedMemPermuteSliceIfRealBatchNotChanged
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x4x6x8xf16>
func.func @SwapBatchedMemPermuteSliceIfRealBatchNotChanged(%arg0: tensor<2x4x6x8xf16>) -> (tensor<1x4x6x8xf16, {order = #CNWH}>, tensor<1x4x6x8xf16, {order = #CNWH}>) {
    %0 = IE.MemPermute(%arg0) {dst_order = #CNWH, mem_perm = #CNWH} : tensor<2x4x6x8xf16> -> tensor<2x4x6x8xf16, {order = #CNWH}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 4, 6, 8] : tensor<2x4x6x8xf16, {order = #CNWH}> to tensor<1x4x6x8xf16, {order = #CNWH}>
    %2 = IE.Slice %0 [1, 0, 0, 0] [1, 4, 6, 8] : tensor<2x4x6x8xf16, {order = #CNWH}> to tensor<1x4x6x8xf16, {order = #CNWH}>

    return %1, %2 : tensor<1x4x6x8xf16, {order = #CNWH}>, tensor<1x4x6x8xf16, {order = #CNWH}>

    // CHECK:       [[SLICE0:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 4, 6, 8] : tensor<2x4x6x8xf16> to tensor<1x4x6x8xf16>
    // CHECK:       [[MEMPERMUTE0:%.*]] = IE.MemPermute([[SLICE0]]) {dst_order = #map{{[0-9]*}}, mem_perm = #map{{[0-9]*}}} : tensor<1x4x6x8xf16> -> tensor<1x4x6x8xf16, {order = #map{{[0-9]*}}}>
    // CHECK:       [[SLICE1:%.*]] = IE.Slice [[INPUT]] [1, 0, 0, 0] [1, 4, 6, 8] : tensor<2x4x6x8xf16> to tensor<1x4x6x8xf16>
    // CHECK:       [[MEMPERMUTE1:%.*]] = IE.MemPermute([[SLICE1]]) {dst_order = #map{{[0-9]*}}, mem_perm = #map{{[0-9]*}}} : tensor<1x4x6x8xf16> -> tensor<1x4x6x8xf16, {order = #map{{[0-9]*}}}>
    // CHECK:       return [[MEMPERMUTE0]], [[MEMPERMUTE1]] : tensor<1x4x6x8xf16, {order = #map{{[0-9]*}}}>, tensor<1x4x6x8xf16, {order = #map{{[0-9]*}}}>
}
