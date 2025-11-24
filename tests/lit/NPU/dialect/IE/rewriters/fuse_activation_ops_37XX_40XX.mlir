//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-activation-ops="enable-fuse-clamp=false" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --run-adjust-for-vpu-rewriters="enable-fuse-clamp=false rewriter=fuse-activation-ops-set" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FakeQuantConv2dWithLeakyRelu1Test
func.func @FakeQuantConv2dWithLeakyRelu1Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %cst = const.Declare tensor<f16> = dense<1.270000e+02> : tensor<f16>
    %cst_0 = const.Declare tensor<f16> = dense<-1.280000e+02> : tensor<f16>
    %cst_1 = const.Declare tensor<f16> = dense<6.000000e+00> : tensor<f16>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
    // CHECK-SAME:     strides = [1, 1]
}

// -----

// CHECK-LABEL: @FakeQuantConv2dWithLeakyRelu15Test
func.func @FakeQuantConv2dWithLeakyRelu15Test(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %cst = const.Declare tensor<f16> = dense<1.270000e+02> : tensor<f16>
    %cst_0 = const.Declare tensor<f16> = dense<-1.280000e+02> : tensor<f16>
    %cst_1 = const.Declare tensor<f16> = dense<6.000000e+00> : tensor<f16>

    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x4x4xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x4x4xf16>

    %0 = IE.Convolution(%quantized_input, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.500000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %result = IE.FakeQuantize(%1, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x16x3x3xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x16x3x3xf16>

    return %result : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-SAME:     post_op = #IE.LeakyRelu<negative_slope = 1.500000e-01 : f64>
    // CHECK-SAME:     strides = [1, 1]
}

// -----

// CHECK-LABEL: @FakeQuantAddWithLeakyReluFusedTest
func.func @FakeQuantAddWithLeakyReluFusedTest(%arg0: tensor<1x128x1x8xf16>) -> tensor<1x128x1x8xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_2, %cst_1, %cst_2, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %2 = IE.Add(%0, %1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %3 = IE.LeakyRelu(%2) {
            negative_slope = 0.199951171875 : f64
    } : tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %4 = IE.FakeQuantize(%3, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    return %4 : tensor<1x128x1x8xf16>

    // CHECK:       IE.Add
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 0.199951171875 : f64>
    // CHECK-SAME:  } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
}

// -----

// CHECK-LABEL: @AvgPoolWithLeakyReluFuseTest
func.func @AvgPoolWithLeakyReluFuseTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.AvgPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<FLOOR>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {
            negative_slope = 1.000000e-01 : f64
        } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[AVG_POOL:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       return [[AVG_POOL]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @SkipFakeQuantAndOpWithLeakyReluFusedTest
func.func @SkipFakeQuantAndOpWithLeakyReluFusedTest(%arg0: tensor<1x128x1x8xf16>) -> tensor<1x128x1x8xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_2, %cst_1, %cst_2, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %2 = IE.And(%0, %1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %3 = IE.LeakyRelu(%2) {
            negative_slope = 1.000000e-01 : f64
    } : tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %4 = IE.FakeQuantize(%3, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    return %4 : tensor<1x128x1x8xf16>

    // CHECK:       IE.And
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    // CHECK:       IE.LeakyRelu
    // CHECK-SAME:      negative_slope = 1.000000e-01 : f64
    // CHECK-SAME:  } : tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
}

// -----

// CHECK-LABEL: @SkipFakeQuantMultiplyWithLeakyReluFusedTest
func.func @SkipFakeQuantMultiplyWithLeakyReluFusedTest(%arg0: tensor<1x128x1x8xf16>) -> tensor<1x128x1x8xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<-6.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_2, %cst_1, %cst_2, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    %2 = IE.Multiply(%0, %1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %3 = IE.LeakyRelu(%2) {
            negative_slope = 2.000000e-01 : f64
    } : tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    %4 = IE.FakeQuantize(%3, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    } : tensor<1x128x1x8xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x1x8xf16>
    return %4 : tensor<1x128x1x8xf16>

    // CHECK:       IE.Multiply
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x128x1x8xf16>, tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
    // CHECK:       IE.LeakyRelu
    // CHECK-SAME:      negative_slope = 2.000000e-01 : f64
    // CHECK-SAME:  } : tensor<1x128x1x8xf16> -> tensor<1x128x1x8xf16>
}

// -----

// CHECK-LABEL: @SkipMaxPoolWithReluTest
func.func @SkipMaxPoolWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<FLOOR>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       [[RELU:%.*]] = IE.ReLU([[MAX_POOL]])
    // CHECK-SAME:  : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       return [[RELU]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @SkipMaxPoolWithLeakyReluTest
func.func @SkipMaxPoolWithLeakyReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2, 2],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<FLOOR>
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.LeakyRelu(%0) {negative_slope = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.*]] = IE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       [[LEAKY_RELU:%.*]] = IE.LeakyRelu([[MAX_POOL]]) {
    // CHECK-SAME:      negative_slope = 1.000000e-01 : f64
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       return [[LEAKY_RELU]] : tensor<1x16x3x3xf16>
}

// -----

// CHECK-LABEL: @Conv2dWithSigmoidNotFusedTest
func.func @Conv2dWithSigmoidNotFusedTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Sigmoid(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-NOT:     post_op = #IE.Sigmoid<>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NEXT:   IE.Sigmoid
}

// -----

// CHECK-LABEL: @Conv2dWithTanhNotFusedTest
func.func @Conv2dWithTanhNotFusedTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = dense<1.0> : tensor<16x16x2x2xf16>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Tanh(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       IE.Convolution
    // CHECK-SAME:     dilations = [1, 1]
    // CHECK-SAME:     pads_begin = [0, 0]
    // CHECK-SAME:     pads_end = [0, 0]
    // CHECK-NOT:     post_op = #IE.Tanh<>
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-NEXT:   IE.Tanh
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK: func.func @MixedAvgPoolWithLeakyReluNoFuseTest([[ARG0:%.+]]: tensor<1x16x4x4x!qElemType>) -> tensor<1x16x4x4xf16> {
func.func @MixedAvgPoolWithLeakyReluNoFuseTest(%arg0: tensor<1x16x4x4x!qElemType>) -> tensor<1x16x4x4xf16> {
    %0 = IE.AvgPool(%arg0)
         {
             kernel_size = [1, 1],
             pads_begin = [0, 0],
             pads_end = [0, 0],
             strides = [1, 1],
             rounding_type = #IE.rounding_type<FLOOR>
         } :
         tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4xf16>

    %1 = IE.LeakyRelu(%0) {
            negative_slope = 1.000000e-01 : f64
        } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    return %1 : tensor<1x16x4x4xf16>

    // CHECK:       [[AVG_POOL:%.*]] = IE.AvgPool([[ARG0]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4xf16>
    // CHECK:       [[LEAKY_RELU:%.*]] = IE.LeakyRelu([[AVG_POOL]]) {
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x4x4xf16>

    // CHECK:       return [[LEAKY_RELU]] : tensor<1x16x4x4xf16>
}

// -----

// CHECK-LABEL: @SkipMaxPoolWithClampTest
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x4x4xf16>)
func.func @SkipMaxPoolWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
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

    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[MAX_POOL]]) {
    // CHECK-SAME:  max = 6.000000e+00 : f64,
    // CHECK-SAME:  min = 0.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK:       return [[CLAMP]] : tensor<1x16x3x3xf16>
}


// -----

// CHECK-LABEL: @NotFuseClampI32
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x50x1x1xsi32>)
func.func @NotFuseClampI32(%arg0: tensor<1x50x1x1xsi32>) -> tensor<1x50x1x1xsi32> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>
    %1 = IE.Clamp(%0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>
    return %1 : tensor<1x50x1x1xsi32>

    // CHECK:       [[AvgPool:%.+]] = IE.AvgPool([[INPUT]]) {
    // CHECK-SAME:      exclude_pads
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>
    // CHECK:       [[Clamp:%.+]] = IE.Clamp([[AvgPool]]) {
    // CHECK-SAME:      max = 1.000000e+00 : f64,
    // CHECK-SAME:      min = -1.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>

}
