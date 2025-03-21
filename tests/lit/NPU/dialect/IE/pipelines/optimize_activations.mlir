//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-activations="enable-fuse-clamp-op=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @OptimizeActivationsConvMaxPoolRelu
func.func @OptimizeActivationsConvMaxPoolRelu(%arg0: tensor<1x16x5x5xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %conv = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x5x5xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x4x4xf16>

    %maxpool = IE.MaxPool(%conv)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<CEIL>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %relu = IE.ReLU(%maxpool) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %relu : tensor<1x16x3x3xf16>

    // CHECK:       [[CST:%.*]] = const.Declare
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]])
    // CHECK-SAME:  post_op = #IE.PostOp<name = "IE.ReLU"
    // CHECK:       [[MAX_POOL:%.*]] = IE.MaxPool([[CONV]])
    // CHECK:       return [[MAX_POOL]]
}


// -----

// CHECK-LABEL: @OptimizeActivationsConv
func.func @OptimizeActivationsConv(%arg0: tensor<4x512x1x1xf16>, %arg1: tensor<4x2048x2x1xf16>) -> tensor<4x2048x3x1xf16> {
   %cst = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
   %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>

    %1 = IE.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<4x2048x1x1xf16>, tensor<4x2048x2x1xf16> -> tensor<4x2048x3x1xf16>
    %2 = IE.Clamp(%1) {max = 0.700000e+00 : f64, min = 0.000000e+00 : f64} : tensor<4x2048x3x1xf16> -> tensor<4x2048x3x1xf16>

    return %2 : tensor<4x2048x3x1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<2048x512x1x1xf16> = dense<1.000000e+00> : tensor<2048x512xf16>, [#const.Reshape<[2048, 512, 1, 1]>]
    // CHECK:       [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:   clamp = {max = 0.69999999999999996 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:   dilations = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:   } : tensor<4x512x1x1xf16>, tensor<2048x512x1x1xf16> -> tensor<4x2048x1x1xf16>
    // CHECK-NOT:   IE.Clamp
    // CHECK:       [[AVG_POOL:%.*]] =  IE.AvgPool(%arg1) {
    // CHECK-SAME:   clamp = {max = 0.69999999999999996 : f64, min = 0.000000e+00 : f64},
    // CHECK-SAME:   exclude_pads,
    // CHECK-SAME:   kernel_size = [1, 1],
    // CHECK-SAME:   pads_begin = [0, 0],
    // CHECK-SAME:   pads_end = [0, 0],
    // CHECK-SAME:   rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:   strides = [1, 1]
    // CHECK-SAME:   } : tensor<4x2048x2x1xf16> -> tensor<4x2048x2x1xf16>
    // CHECK-NOT:   IE.Clamp
    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[CONV]], [[AVG_POOL]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<4x2048x1x1xf16>, tensor<4x2048x2x1xf16> -> tensor<4x2048x3x1xf16>
    // CHECK:       return [[CONCAT]] : tensor<4x2048x3x1xf16>
}


// -----

// CHECK-LABEL: @OptimizeActivationsAddWithMultipleUsers
func.func @OptimizeActivationsAddWithMultipleUsers(%arg0: tensor<1x32x4x4xf16>, %arg1: tensor<1x128x4x4xf16>, %arg2: tensor<1x128x4x4xf16>) -> (tensor<1x128x8x2xf16>, tensor<1x128x4x4xf16>) {
    %cst = const.Declare tensor<128x32x1x1xf16> = dense<1.000000e+00> : tensor<128x32xf16>, [#const.Reshape<[128, 32, 1, 1]>]
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<128x32x1x1xf16> -> tensor<1x128x4x4xf16>
    %add0 = IE.Add(%conv, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %reshape = IE.Reshape(%add0) {shape_value = [1, 128, 8, 2]} : tensor<1x128x4x4xf16> -> tensor<1x128x8x2xf16>
    %relu = IE.ReLU(%reshape) : tensor<1x128x8x2xf16> -> tensor<1x128x8x2xf16>
    %add1 = IE.Add(%add0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %relu, %add1 : tensor<1x128x8x2xf16>, tensor<1x128x4x4xf16>

    // CHECK:    [[CST:%.*]] = const.Declare tensor<128x32x1x1xf16> = dense<1.000000e+00> : tensor<128x32xf16>, [#const.Reshape<[128, 32, 1, 1]>]
    // CHECK:    [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME: dilations = [1, 1],
    // CHECK-SAME: pads_begin = [0, 0],
    // CHECK-SAME: pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME: } : tensor<1x32x4x4xf16>, tensor<128x32x1x1xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[ADD0:%.*]] = IE.Add([[CONV]], %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[AVGPOOL:%.*]] = IE.AvgPool([[ADD0]]) {
    // CHECK-SAME: exclude_pads,
    // CHECK-SAME: kernel_size = [1, 1],
    // CHECK-SAME: pads_begin = [0, 0],
    // CHECK-SAME: pads_end = [0, 0],
    // CHECK-SAME: post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>
    // CHECK-SAME: rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME: strides = [1, 1]
    // CHECK-SAME: } : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[RESHAPE:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 128, 8, 2]} : tensor<1x128x4x4xf16> -> tensor<1x128x8x2xf16>
    // CHECK:    [[ADD1:%.*]] = IE.Add([[ADD0]], %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    // CHECK:    return [[RESHAPE]], [[ADD1]] : tensor<1x128x8x2xf16>, tensor<1x128x4x4xf16>
}

// -----

// CHECK-LABEL: @OptimizeActivationsGroupConv
func.func @OptimizeActivationsGroupConv(%arg0: tensor<2x512x32x48xf16>,
                                        %arg1: tensor<2x512x32x48xf16>)
        -> tensor<2x512x64x48xf16> {
    %weights = const.Declare tensor<512x1x3x3xf16> = dense<1.000000e+00> : tensor<512x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %weights) {
        dilations = [1, 1],
        groups = 512 : i64,
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<2x512x32x48xf16>, tensor<512x1x3x3xf16> -> tensor<2x512x32x48xf16>

    %1 = IE.Concat(%0, %arg1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]
    } : tensor<2x512x32x48xf16>, tensor<2x512x32x48xf16> -> tensor<2x512x64x48xf16>

    %2 = IE.ReLU(%1) : tensor<2x512x64x48xf16> -> tensor<2x512x64x48xf16>

    return %2 : tensor<2x512x64x48xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<512x1x3x3xf16> = dense<1.000000e+00>
    // CHECK:   [[DWCONV:%.*]] = IE.GroupConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 512 : i64,
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<2x512x32x48xf16>, tensor<512x1x3x3xf16> -> tensor<2x512x32x48xf16>

    // CHECK-NOT:   IE.ReLU
    // CHECK:   [[AVG_POOL:%.*]] =  IE.AvgPool(%arg1) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<2x512x32x48xf16> -> tensor<2x512x32x48xf16>

    // CHECK-NOT:   IE.ReLU
    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[DWCONV]], [[AVG_POOL]]) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 32, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<2x512x32x48xf16>, tensor<2x512x32x48xf16> -> tensor<2x512x64x48xf16>

    // CHECK:       return [[CONCAT]] : tensor<2x512x64x48xf16>
}
