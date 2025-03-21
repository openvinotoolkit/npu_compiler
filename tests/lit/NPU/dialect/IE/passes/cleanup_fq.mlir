//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --cleanup-fq %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotCleanupLastFQ([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotCleanupLastFQ(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x28x70xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]]
    // CHECK: return [[FQ]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotCleanupFirstFQ([[ARG0:%.+]]: tensor<1x70x1x28xf16>)
func.func @DoNotCleanupFirstFQ(%arg0: tensor<1x70x1x28xf16>) -> tensor<1x1x28x70xf16> {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %0 = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %1 = IE.Transpose(%0) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %1 : tensor<1x1x28x70xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]])
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x28x70xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @CleanupPerTensorFQ([[ARG0:%.+]]: tensor<1x70x2x14xf16>)
func.func @CleanupPerTensorFQ(%arg0: tensor<1x70x2x14xf16>) -> tensor<1x1x28x70xf16> {
    %inLow = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %inHigh = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    %outLow = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %outHigh = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    %0 = IE.Reshape(%arg0) { shape_value = [1, 70, 1, 28] } : tensor<1x70x2x14xf16> -> tensor<1x70x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x70x1x28xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x70x1x28xf16>
    %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x70x1x28xf16> -> tensor<1x1x28x70xf16>
    return %2 : tensor<1x1x28x70xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK-NOT: FakeQuantize
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK: return [[TRANSPOSE]] : tensor<1x1x28x70xf16>
}

// -----

// CHECK: func.func @CleanupPerAxisFQ([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @CleanupPerAxisFQ(%arg0: tensor<4x1x2x14xf16>) -> tensor<4x1x1x56xf16> {
    %inLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094492]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %inHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257814]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>
    %outLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %outHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>

    %0 = IE.Reshape(%arg0) { shape_value = [4, 1, 1, 28] } : tensor<4x1x2x14xf16> -> tensor<4x1x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x1x28xf16>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x1x1x28xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x56xf16>
    return %2 : tensor<4x1x1x56xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK-NOT: FakeQuantize
    // CHECK: [[TILE:%.+]] = IE.Tile([[RESHAPE]])
    // CHECK: return [[TILE]] : tensor<4x1x1x56xf16>
}

// -----

// CHECK: func.func @CleanupMultipleConsumersFQ([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @CleanupMultipleConsumersFQ(%arg0: tensor<4x1x2x14xf16>) -> (tensor<4x1x1x56xf16>, tensor<8x1x1x28xf16>) {
    %inLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094492]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %inHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257814]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>
    %outLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %outHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>

    %0 = IE.Reshape(%arg0) { shape_value = [4, 1, 1, 28] } : tensor<4x1x2x14xf16> -> tensor<4x1x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x1x28xf16>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x1x1x28xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x56xf16>
    %3 = IE.Tile(%1) {repeats_values = [2, 1, 1, 1]} : tensor<4x1x1x28xf16> -> tensor<8x1x1x28xf16>
    return %2, %3 : tensor<4x1x1x56xf16>, tensor<8x1x1x28xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK-NOT: FakeQuantize
    // CHECK: [[TILE1:%.+]] = IE.Tile([[RESHAPE]])
    // CHECK: [[TILE2:%.+]] = IE.Tile([[RESHAPE]])
    // CHECK: return [[TILE1]], [[TILE2]] : tensor<4x1x1x56xf16>, tensor<8x1x1x28xf16>
}

// -----

// CHECK: func.func @DoNotCleanupPerAxisFQDifferentValues([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @DoNotCleanupPerAxisFQDifferentValues(%arg0: tensor<4x1x2x14xf16>) -> tensor<4x1x1x56xf16> {
    %inLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.1]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %inHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257814]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>
    %outLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.2]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %outHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>

    %0 = IE.Reshape(%arg0) { shape_value = [4, 1, 1, 28] } : tensor<4x1x2x14xf16> -> tensor<4x1x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x1x28xf16>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x1x1x28xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x56xf16>
    return %2 : tensor<4x1x1x56xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE]]
    // CHECK: [[TILE:%.+]] = IE.Tile([[FQ]])
    // CHECK: return [[TILE]] : tensor<4x1x1x56xf16>
}

// -----

// CHECK: func.func @DoNotCleanupPerAxisFQDifferentSizes([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @DoNotCleanupPerAxisFQDifferentSizes(%arg0: tensor<4x1x2x14xf16>) -> tensor<4x1x1x56xf16> {
    %inLow = const.Declare tensor<f32> = dense<-0.619094491> : tensor<f32>
    %inHigh = const.Declare tensor<f32> = dense<0.614257813> : tensor<f32>
    %outLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %outHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>

    %0 = IE.Reshape(%arg0) { shape_value = [4, 1, 1, 28] } : tensor<4x1x2x14xf16> -> tensor<4x1x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x1x28xf16>, tensor<f32>, tensor<f32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x1x1x28xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x56xf16>
    return %2 : tensor<4x1x1x56xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE]]
    // CHECK: [[TILE:%.+]] = IE.Tile([[FQ]])
    // CHECK: return [[TILE]] : tensor<4x1x1x56xf16>
}

// -----

// CHECK: func.func @DoNotCleanupPerAxisFQAndMaxPool([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @DoNotCleanupPerAxisFQAndMaxPool(%arg0: tensor<4x1x2x14xf16>) -> (tensor<4x1x1x56xf16>, tensor<4x1x1x28xf16>) {
    %inLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094492]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %inHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257814]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>
    %outLow = const.Declare tensor<4x1x1x1xf32> = dense<[[[[-0.619094491]]], [[[-0.0362020172]]], [[[-0.517224431]]], [[[-0.00619094493]]]]> : tensor<4x1x1x1xf32>
    %outHigh = const.Declare tensor<4x1x1x1xf32> = dense<[[[[0.614257813]]], [[[0.0359191895]]], [[[0.513183594]]], [[[0.00614257809]]]]> : tensor<4x1x1x1xf32>

    %0 = IE.Reshape(%arg0) { shape_value = [4, 1, 1, 28] } : tensor<4x1x2x14xf16> -> tensor<4x1x1x28xf16>
    %1 = IE.FakeQuantize(%0, %inLow, %inHigh, %outLow, %outHigh)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x1x28xf16>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32> -> tensor<4x1x1x28xf16>
    %2 = IE.Tile(%1) {repeats_values = [1, 1, 1, 2]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x56xf16>
    %3 = IE.MaxPool(%1) {exclude_pads, kernel_size = [1, 1],
                            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
                            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<4x1x1x28xf16> -> tensor<4x1x1x28xf16>
    return %2, %3 : tensor<4x1x1x56xf16>, tensor<4x1x1x28xf16>

    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[RESHAPE]]
    // CHECK: [[TILE:%.+]] = IE.Tile([[FQ]])
    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[FQ]])
    // CHECK: return [[TILE]], [[MAX_POOL]] : tensor<4x1x1x56xf16>, tensor<4x1x1x28xf16>
}

// -----

// CHECK: func.func @DoNotCleanupMaxPoolAndFQ([[ARG0:%.+]]: tensor<4x1x2x14xf16>)
func.func @DoNotCleanupMaxPoolAndFQ(%arg0: tensor<4x1x2x14xf16>) -> tensor<4x1x2x14xf16> {
    %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>

    %0 = IE.MaxPool(%arg0) {exclude_pads, kernel_size = [1, 1],
                            pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.PostOp<name = "IE.LeakyRelu",
                            attrs = {negative_slope = 0.10000000149011612 : f64}>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<4x1x2x14xf16> -> tensor<4x1x2x14xf16>
    %1 = IE.FakeQuantize(%0, %cst, %cst_0, %cst, %cst_0)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
                    : tensor<4x1x2x14xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<4x1x2x14xf16>
    return %1 : tensor<4x1x2x14xf16>

    // CHECK: [[MAX_POOL:%.+]] = IE.MaxPool([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[MAX_POOL]]
    // CHECK: return [[FQ]] : tensor<4x1x2x14xf16>
}
