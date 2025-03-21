//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --insert-identity-pool-before-op %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @SkipLReluWithNCEProducer
func.func @SkipLReluWithNCEProducer(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%MAX_POOL) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %LRELU : tensor<1x128x2x32xf16>

    // CHECK:   [[MAX_POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[MAX_POOL]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[LRELU]] : tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolBetweenMaxPoolAndClamp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x4x4xf16>)
func.func @InsertAvgPoolBetweenMaxPoolAndClamp(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<FLOOR>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00,
            min = 0.000000e+00
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[ARG0]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       [[AVG_POOL:%.+]] = IE.AvgPool([[MAX_POOL]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[AVG_POOL]]) {
    // CHECK-SAME:  max = 6.000000e+00 : f64,
    // CHECK-SAME:  min = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       return [[CLAMP]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToConcatAndLRelu
func.func @InsertAvgPoolToConcatAndLRelu(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToConcatAndClamp
func.func @InsertAvgPoolToConcatAndClamp(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToSplitAndLRelu
func.func @InsertAvgPoolToSplitAndLRelu(%arg0: tensor<1x128x2x32xf16>) -> (tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>) {
    %0:2 = IE.Split(%arg0) {axis_value = 1, num_splits = 2} : tensor<1x128x2x32xf16> -> tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
    %1 = IE.LeakyRelu(%0#0) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    %2 = IE.LeakyRelu(%0#1) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>

    return %1, %2 : tensor<1x64x2x32xf16> , tensor<1x64x2x32xf16>

    // CHECK:   %[[VAL_0:.*]]:2 = IE.Split(%arg0) {axis_value = 1 : i64, num_splits = 2 : i64} : tensor<1x128x2x32xf16> -> tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]#0) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_3:.*]] = IE.AvgPool(%[[VAL_0]]#1) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>
    // CHECK:   %[[VAL_4:.*]] = IE.LeakyRelu(%[[VAL_3]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x64x2x32xf16> -> tensor<1x64x2x32xf16>

    // CHECK:   return %[[VAL_2]], %[[VAL_4]] : tensor<1x64x2x32xf16>, tensor<1x64x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToLReluWhenProducerHasPost
func.func @InsertAvgPoolToLReluWhenProducerHasPost(%arg0: tensor<1x25x135x240xf16>) -> tensor<1x188x135x240xf16> {
    %cst = const.Declare tensor<188x25x3x3xf16> = dense<7.558590e-01> : tensor<188x25x3x3xf16>
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1],
            post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.199951171875 : f64}>, strides = [1, 1]} :
            tensor<1x25x135x240xf16>, tensor<188x25x3x3xf16> -> tensor<1x188x135x240xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.199951171875 : f64} : tensor<1x188x135x240xf16> -> tensor<1x188x135x240xf16>

    return %1 : tensor<1x188x135x240xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<188x25x3x3xf16> = dense<7.558590e-01> : tensor<188x25x3x3xf16>
    // CHECK:   [[CONV:%.*]] = IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:          post_op = #IE.PostOp<name = "IE.LeakyRelu", attrs = {negative_slope = 0.199951171875 : f64}>, strides = [1, 1]} :
    // CHECK-SAME:          tensor<1x25x135x240xf16>, tensor<188x25x3x3xf16> -> tensor<1x188x135x240xf16>
    // CHECK:   [[AVG:%.*]] = IE.AvgPool([[CONV]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} :
    // CHECK-SAME:          tensor<1x188x135x240xf16> -> tensor<1x188x135x240xf16>
    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[AVG]]) {negative_slope = 0.199951171875 : f64} : tensor<1x188x135x240xf16> -> tensor<1x188x135x240xf16
    // CHECK:   return [[LRELU]] : tensor<1x188x135x240xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToReshapeAndClamp
func.func @InsertAvgPoolToReshapeAndClamp(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x2x32x128xf16> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 2, 32, 128]} : tensor<1x128x2x32xf16> -> tensor<1x2x32x128xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>

    return %1 : tensor<1x2x32x128xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 32, 128]} : tensor<1x128x2x32xf16> -> tensor<1x2x32x128xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x2x32x128xf16> -> tensor<1x2x32x128xf16>

    // CHECK:   return %[[VAL_2]] : tensor<1x2x32x128xf16>
}

// -----

// CHECK-LABEL: @DoNotInsertAvgPoolToAddAndClamp
func.func @DoNotInsertAvgPoolToAddAndClamp(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    return %1 : tensor<1x128x2x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.Clamp(%[[VAL_0]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>
    // CHECK:   return %[[VAL_1]] : tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToFakeQuantizeAndLeakyRelu
func.func @InsertAvgPoolToFakeQuantizeAndLeakyRelu(%arg0: tensor<1x16x8x8xf16>) -> tensor<1x16x8x8xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<  1x16x8x8xf16>
    %2 = IE.LeakyRelu(%1) { negative_slope = 1.000000e-01 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>

    return %2 : tensor<1x16x8x8xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<7.558590e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[CST_0:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:     [[FQ:%.*]] = IE.FakeQuantize(%arg0, [[CST_0]], [[CST]], [[CST_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x8x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x8x8xf16>
    // CHECK:     [[AVG_POOL:%.*]] = IE.AvgPool([[FQ]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>
    // CHECK:     [[LEAKY_RELU:%.*]] = IE.LeakyRelu([[AVG_POOL]]) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x8x8xf16> -> tensor<1x16x8x8xf16>

    //CHECK:  return [[LEAKY_RELU]] : tensor<1x16x8x8xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @InsertMaxPoolBeforeMemPermute
func.func @InsertMaxPoolBeforeMemPermute(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>)
    -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[POOLING:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InsertAvgPoolToAddWithMultipleUsersAndRelu
func.func @InsertAvgPoolToAddWithMultipleUsersAndRelu(%arg0: tensor<1x128x4x4xf16>, %arg1: tensor<1x128x4x4xf16>) -> tensor<1x128x4x4xf16> {
    %add0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %relu = IE.ReLU(%add0) : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    %add1 = IE.Add(%add0, %relu) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %add1 : tensor<1x128x4x4xf16>

    // CHECK:    [[ADD0:%.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[AVGPOOL:%.*]] = IE.AvgPool([[ADD0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[RELU:%.*]] = IE.ReLU([[AVGPOOL]]) : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[ADD1:%.*]] = IE.Add([[ADD0]], [[RELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4xf16>, tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    // CHECK:   return [[ADD1]] : tensor<1x128x4x4xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipMemPermuteWithNCE
func.func @SkipMemPermuteWithNCE(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>)
    -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %POOLING = IE.MaxPool(%arg0) {
          kernel_size = [5, 5],
          pads_begin = [2, 2],
          pads_end = [2, 2],
          rounding_type = #IE.rounding_type<FLOOR>,
          strides = [1, 1]
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x16x32x64xf16, {order = #NHWC}>

    %PERMUTE = IE.MemPermute(%POOLING) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[POOLING:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [5, 5],
    // CHECK-SAME:      pads_begin = [2, 2],
    // CHECK-SAME:      pads_end = [2, 2],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @SkipMemPermuteWithNCHWInput
func.func @SkipMemPermuteWithNCHWInput(%arg0: tensor<1x16x32x64xf16>)
    -> tensor<1x32x16x64xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NCWH
    } : tensor<1x16x32x64xf16> -> tensor<1x32x16x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NCWH
    // CHECK-SAME:  } : tensor<1x16x32x64xf16>
    // CHECK-SAME:      -> tensor<1x32x16x64xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipMemPermuteWithMisalignedChannels
func.func @SkipMemPermuteWithMisalignedChannels(%arg0: tensor<1x3x32x64xf16, {order = #NHWC}>)
    -> tensor<1x32x3x64xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x3x32x64xf16, {order = #NHWC}> -> tensor<1x32x3x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x3x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x3x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x3x64xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x3x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipMemPermuteWithBatch
func.func @SkipMemPermuteWithBatch(%arg0: tensor<2x16x32x64xf16, {order = #NHWC}>)
    -> tensor<2x32x16x64xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<2x16x32x64xf16, {order = #NHWC}> -> tensor<2x32x16x64xf16, {order = #NHWC}>

    return %PERMUTE : tensor<2x32x16x64xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<2x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<2x32x16x64xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<2x32x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @SkipMemPermuteWithUnsupportedType
func.func @SkipMemPermuteWithUnsupportedType(%arg0: tensor<1x16x32x64xsi32, {order = #NHWC}>)
    -> tensor<1x32x16x64xsi32, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x64xsi32, {order = #NHWC}> -> tensor<1x32x16x64xsi32, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x64xsi32, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:      } : tensor<1x16x32x64xsi32, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x16x64xsi32, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x16x64xsi32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipLReluWithNCEProducer
func.func @SkipLReluWithNCEProducer(%arg0: tensor<1x128x2x32xf16>) -> tensor<1x128x2x32xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%AVG_POOL) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %LRELU : tensor<1x128x2x32xf16>

    // CHECK:   [[AVG_POOL:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[AVG_POOL]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[LRELU]] : tensor<1x128x2x32xf16>
}

// -----

// CHECK-LABEL: @LReluWithNCEProducerAndTwoUsers
func.func @LReluWithNCEProducerAndTwoUsers(%arg0: tensor<1x128x2x32xf16>)
        -> (tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>) {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    %LRELU = IE.LeakyRelu(%MAX_POOL) {
        negative_slope = 0.000000e+00 : f64
    } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    return %MAX_POOL, %LRELU : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>

    // CHECK:   [[ORIG_POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[NEW_POOL:%.*]] = IE.AvgPool([[ORIG_POOL]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   [[LRELU:%.*]] = IE.LeakyRelu([[NEW_POOL]]) {
    // CHECK-SAME:      negative_slope = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x128x2x32xf16> -> tensor<1x128x2x32xf16>

    // CHECK:   return [[ORIG_POOL]], [[LRELU]] : tensor<1x128x2x32xf16>, tensor<1x128x2x32xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>

// CHECK-LABEL: @SkipMemPermuteWithUnsupportedOrder
func.func @SkipMemPermuteWithUnsupportedOrder(%arg0: tensor<2x16x32x64xf16, {order = #NHWC}>)
    -> tensor<16x2x64x32xf16, {order = #NHWC}> {

    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #map
    } : tensor<2x16x32x64xf16, {order = #NHWC}> -> tensor<16x2x64x32xf16, {order = #NHWC}>

    return %PERMUTE : tensor<16x2x64x32xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #map
    // CHECK-SAME:  } : tensor<2x16x32x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<16x2x64x32xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<16x2x64x32xf16, {order = #NHWC}>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @SkipMemPermuteWithSmallHeight
func.func @SkipMemPermuteWithSmallHeight(%arg0: tensor<1x16x1x64xf16, {order = #NHWC}>)
    -> tensor<1x16x1x64xf16> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NCHW,
        mem_perm = #NWCH
    } : tensor<1x16x1x64xf16, {order = #NHWC}> -> tensor<1x16x1x64xf16>

    return %PERMUTE : tensor<1x16x1x64xf16>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute(%arg0) {
    // CHECK-SAME:      dst_order = #NCHW,
    // CHECK-SAME:      mem_perm = #NWCH
    // CHECK-SAME:  } : tensor<1x16x1x64xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x1x64xf16>

    // CHECK:   return [[PERMUTE]] : tensor<1x16x1x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL: @MemPermuteWithSmallByteSize
func.func @MemPermuteWithSmallByteSize(%arg0: tensor<1x16x32x2xf16, {order = #NHWC}>)
    -> tensor<1x32x16x2xf16, {order = #NHWC}> {
    %PERMUTE = IE.MemPermute(%arg0) {
        dst_order = #NHWC,
        mem_perm = #NWHC
    } : tensor<1x16x32x2xf16, {order = #NHWC}> -> tensor<1x32x16x2xf16, {order = #NHWC}>

    return %PERMUTE : tensor<1x32x16x2xf16, {order = #NHWC}>

    // CHECK:   [[POOLING:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x32x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x32x2xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTE:%.*]] = IE.MemPermute([[POOLING]]) {
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NWHC
    // CHECK-SAME:  } : tensor<1x16x32x2xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x32x16x2xf16, {order = #NHWC}>

    // CHECK:   return [[PERMUTE]] : tensor<1x32x16x2xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @InsertAvgPoolIfActivationHasBatch
func.func @InsertAvgPoolIfActivationHasBatch(%arg0: tensor<1x12x10x20xf16>) -> tensor<3x4x10x20xf16> {
    %reshape = IE.Reshape(%arg0) {shape_value = [3, 4, 10, 20]} : tensor<1x12x10x20xf16> -> tensor<3x4x10x20xf16>
    %relu = IE.ReLU(%reshape) : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>
    return %relu : tensor<3x4x10x20xf16>

    // CHECK:   [[RESHAPE:%.*]] = IE.Reshape(%arg0) {
    // CHECK-SAME:      shape_value = [3, 4, 10, 20]
    // CHECK-SAME:  } : tensor<1x12x10x20xf16> -> tensor<3x4x10x20xf16>

    // CHECK:   [[AVG_POOL:%.*]] = IE.AvgPool([[RESHAPE]]) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>

    // CHECK:   [[RELU:%.*]] = IE.ReLU([[AVG_POOL]]) : tensor<3x4x10x20xf16> -> tensor<3x4x10x20xf16>
    // CHECK:   return [[RELU]] : tensor<3x4x10x20xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 5.000000e-01>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SkipMemPermuteWithNCEAcrossQuantizeCast
func.func @SkipMemPermuteWithNCEAcrossQuantizeCast(%arg0: tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x64x64x16x!qElemType, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType1, {order = #NHWC}>
    %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x16x64x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType, {order = #NHWC}>
    %2 = IE.MemPermute(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x64x!qElemType, {order = #NHWC}> -> tensor<1x64x64x16x!qElemType, {order = #NHWC}>

    return %2 : tensor<1x64x64x16x!qElemType, {order = #NHWC}>

    // CHECK:       [[ADD:%.*]] = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x64x64xf16, {order = #NHWC}>, tensor<1x16x64x64xf16, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType1, {order = #NHWC}>
    // CHECK:       [[QCAST:%.*]] = IE.QuantizeCast([[ADD]]) {dstElemType = !qElemType} : tensor<1x16x64x64x!qElemType1, {order = #NHWC}> -> tensor<1x16x64x64x!qElemType, {order = #NHWC}>
    // CHECK:       [[MEM_PERM:%.*]] = IE.MemPermute([[QCAST]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x16x64x64x!qElemType, {order = #NHWC}> -> tensor<1x64x64x16x!qElemType, {order = #NHWC}>
    // CHECK:       return [[MEM_PERM]] : tensor<1x64x64x16x!qElemType, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK: func.func @InsertAvgPoolToMixedPrecisionAddAndLeakyRelu([[ARG0:%.+]]: tensor<1x128x4x4x!qElemType>) -> tensor<1x128x4x4xf16> {
func.func @InsertAvgPoolToMixedPrecisionAddAndLeakyRelu(%arg0: tensor<1x128x4x4x!qElemType>) -> tensor<1x128x4x4xf16> {
    %ADD = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4x!qElemType>, tensor<1x128x4x4x!qElemType> -> tensor<1x128x4x4xf16>
    %LRELU = IE.LeakyRelu(%ADD) {negative_slope = 0.0999755859375 : f64} : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    return %LRELU : tensor<1x128x4x4xf16>

    // CHECK:    [[ADD0:%.*]] = IE.Add([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x4x!qElemType>, tensor<1x128x4x4x!qElemType> -> tensor<1x128x4x4xf16>
    // CHECK:    [[AVGPOOL:%.*]] = IE.AvgPool([[ADD0]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>
    // CHECK:    [[LEAKY_RELU:%.*]] = IE.LeakyRelu([[AVGPOOL]]) {negative_slope = 0.0999755859375 : f64} : tensor<1x128x4x4xf16> -> tensor<1x128x4x4xf16>

    // CHECK:   return [[LEAKY_RELU]] : tensor<1x128x4x4xf16>
}
