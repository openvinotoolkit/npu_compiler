//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% enable-sprlut=true" --swap-operations %s | FileCheck %s
// REQUIRES: platform-NPU5010 || platform-NPU6010

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithExp
func.func @SwapWithExp(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Exp(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK: IE.Convolution
    // CHECK-SAME: tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK: IE.Exp
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @OptimizeSigmoidReorderNHWC
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x34x34xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<16x8x3x3xf16, {order = #NHWC}>)
func.func @OptimizeSigmoidReorderNHWC(%arg0: tensor<1x8x34x34xf16, {order = #NHWC}>, %arg1: tensor<16x8x3x3xf16, {order = #NHWC}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x34x34xf16, {order = #NHWC}>, tensor<16x8x3x3xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
   %2 = IE.Sigmoid(%1) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %2 : tensor<1x16x32x32xf16>

   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x34x34xf16, {order = #NHWC}>, tensor<16x8x3x3xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[CONV]]) : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>
   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[SIGMOID]]) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16>
}

// -----

// CHECK-LABEL: @SwapWithAffineReshapeAndExp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x25x6xf16>, [[ARG1:%.+]]: tensor<40x16x3x3xf16>)
func.func @SwapWithAffineReshapeAndExp(%arg0: tensor<1x16x25x6xf16>, %arg1: tensor<40x16x3x3xf16>) -> tensor<1x1x920x4xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x25x6xf16>, tensor<40x16x3x3xf16> -> tensor<1x40x23x4xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>
   %2 = IE.Exp(%1) : tensor<1x1x920x4xf16> -> tensor<1x1x920x4xf16>

   return %2 : tensor<1x1x920x4xf16>

  // CHECK: [[CONV:%.+]] =  IE.Convolution([[ARG0]], [[ARG1]])
  // CHECK-SAME{LITERAL}:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x25x6xf16>, tensor<40x16x3x3xf16> -> tensor<1x40x23x4xf16>
  // CHECK: [[VAL0:%.+]] =  IE.Exp([[CONV]]) : tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>
  // CHECK: [[VAL1:%.+]] =  IE.AffineReshape([[VAL0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>

  // CHECK: return [[VAL1]] :  tensor<1x1x920x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithTransposeAndExp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x42x25xf16>, [[ARG1:%.+]]: tensor<4x16x3x3xf16>)
func.func @SwapWithTransposeAndExp(%arg0: tensor<1x16x42x25xf16>, %arg1: tensor<4x16x3x3xf16>) -> tensor<1x40x23x4xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x42x25xf16>, tensor<4x16x3x3xf16> -> tensor<1x4x40x23xf16>
   %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>
   %2 = IE.Exp(%1) :  tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>

   return %2 : tensor<1x40x23x4xf16>

  // CHECK: [[CONV:%.+]] =  IE.Convolution([[ARG0]], [[ARG1]])
  // CHECK-SAME{LITERAL}:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x42x25xf16>, tensor<4x16x3x3xf16> -> tensor<1x4x40x23xf16>
  // CHECK: [[VAL0:%.+]] =  IE.Exp([[CONV]]) : tensor<1x4x40x23xf16> -> tensor<1x4x40x23xf16>
  // CHECK: [[VAL1:%.+]] =  IE.Transpose([[VAL0]])
  // CHECK-SAME{LITERAL}:   {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>

  // CHECK: return [[VAL1]] :  tensor<1x40x23x4xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: @OptimizeSigmoidReorderNHCW
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x8x34x34xf16, {order = #NHCW}>, [[ARG1:%.+]]: tensor<16x8x3x3xf16, {order = #NHCW}>)
func.func @OptimizeSigmoidReorderNHCW(%arg0: tensor<1x8x34x34xf16, {order = #NHCW}>, %arg1: tensor<16x8x3x3xf16, {order = #NHCW}>) -> tensor<1x16x32x32xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x34x34xf16, {order = #NHCW}>, tensor<16x8x3x3xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16, {order = #NHCW}>
   %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
   %2 = IE.Sigmoid(%1) : tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
   return %2 : tensor<1x16x32x32xf16>

   // CHECK:        [[CONV:%.+]] = IE.Convolution([[ARG0]], [[ARG1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x8x34x34xf16, {order = #NHCW}>, tensor<16x8x3x3xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16, {order = #NHCW}>
   // CHECK:        [[SIGMOID:%.+]] = IE.Sigmoid([[CONV]]) : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16, {order = #NHCW}>
   // CHECK:        [[REORDER:%.+]] = IE.Reorder([[SIGMOID]])  {dstOrder = #NCHW} : tensor<1x16x32x32xf16, {order = #NHCW}> -> tensor<1x16x32x32xf16>
}

// -----

// CHECK-LABEL: @SwapWithAffineReshapeAndExp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x25x6xf16>, [[ARG1:%.+]]: tensor<40x16x3x3xf16>)
func.func @SwapWithAffineReshapeAndExp(%arg0: tensor<1x16x25x6xf16>, %arg1: tensor<40x16x3x3xf16>) -> tensor<1x1x920x4xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x25x6xf16>, tensor<40x16x3x3xf16> -> tensor<1x40x23x4xf16>
   %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>
   %2 = IE.Exp(%1) : tensor<1x1x920x4xf16> -> tensor<1x1x920x4xf16>

   return %2 : tensor<1x1x920x4xf16>

  // CHECK: [[CONV:%.+]] =     IE.Convolution([[ARG0]], [[ARG1]])
  // CHECK-SAME{LITERAL}:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x25x6xf16>, tensor<40x16x3x3xf16> -> tensor<1x40x23x4xf16>
  // CHECK: [[EXP:%.+]] =      IE.Exp([[CONV]]) : tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>
  // CHECK: [[RESHAPE:%.+]] =  IE.AffineReshape([[EXP]])
  // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 920, 4]} : tensor<1x40x23x4xf16> -> tensor<1x1x920x4xf16>

  // CHECK: return [[RESHAPE]] :  tensor<1x1x920x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SwapWithTransposeAndExp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x16x42x25xf16>, [[ARG1:%.+]]: tensor<4x16x3x3xf16>)
func.func @SwapWithTransposeAndExp(%arg0: tensor<1x16x42x25xf16>, %arg1: tensor<4x16x3x3xf16>) -> tensor<1x40x23x4xf16> {
   %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x42x25xf16>, tensor<4x16x3x3xf16> -> tensor<1x4x40x23xf16>
   %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>
   %2 = IE.Exp(%1) :  tensor<1x40x23x4xf16> -> tensor<1x40x23x4xf16>

   return %2 : tensor<1x40x23x4xf16>

  // CHECK: [[CONV:%.+]] =       IE.Convolution([[ARG0]], [[ARG1]])
  // CHECK-SAME{LITERAL}:        {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x42x25xf16>, tensor<4x16x3x3xf16> -> tensor<1x4x40x23xf16>
  // CHECK: [[EXP:%.+]] =        IE.Exp([[CONV]]) : tensor<1x4x40x23xf16> -> tensor<1x4x40x23xf16>
  // CHECK: [[TRANSPOSE:%.+]] =  IE.Transpose([[EXP]])
  // CHECK-SAME{LITERAL}:        {order_value = #NHWC} : tensor<1x4x40x23xf16> -> tensor<1x40x23x4xf16>

  // CHECK: return [[TRANSPOSE]] :  tensor<1x40x23x4xf16>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @SwapWithSwish
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<4x512x1x1xf16>)
func.func @SwapWithSwish(%arg0: tensor<4x512x1x1xf16>) -> tensor<1x2048x4x1xf16> {
    %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3], [3]], shape_value = [1, 4, 2048, 1]} : tensor<4x2048x1x1xf16> -> tensor<1x4x2048x1xf16>
    %2 = IE.Transpose(%1) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    %3 = IE.Swish(%2) : tensor<1x2048x4x1xf16> -> tensor<1x2048x4x1xf16>

    return %3 : tensor<1x2048x4x1xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]

    // CHECK:     [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK:     [[SWISH:%.+]] = IE.Swish([[CONV]]) : tensor<4x2048x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK:     [[RESHAPE:%.+]] = IE.AffineReshape([[SWISH]])
    // CHECK:     [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]]) {order_value = #NHCW} : tensor<1x4x2048x1xf16> -> tensor<1x2048x4x1xf16>
    // CHECK:     return [[TRANSPOSE]] : tensor<1x2048x4x1xf16>
}
