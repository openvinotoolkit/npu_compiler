//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW enable-sprlut=true" --fuse-activation-ops="enable-fuse-clamp=false" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW enable-sprlut=true" --run-adjust-for-vpu-rewriters="enable-fuse-clamp=false rewriter=fuse-activation-ops-set" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

// CHECK-LABEL: @FuseMaxPoolWithReluTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
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

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Relu<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.ReLU
}

// -----

// CHECK-LABEL: @FuseMaxPoolWithLeakyReluTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithLeakyReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
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

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.LeakyRelu

}

// -----

// CHECK-LABEL: @FuseMaxPoolWithClampTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithClampTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
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
            max = 6.000000e+00 : f64,
            min = 0.000000e+00 : f64
        } :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Clamp
}

// -----

!qElemType = !quant.uniform<u8:f16, 12.695739985447304:118>

// CHECK-LABEL: func.func @FuseQuantMaxPoolWithClampTest
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x97x177x!qElemType>
func.func @FuseQuantMaxPoolWithClampTest(%arg0: tensor<1x32x97x177x!qElemType>) -> tensor<1x32x96x176x!qElemType> {
    %0 = IE.MaxPool(%arg0)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } :
        tensor<1x32x97x177x!qElemType> -> tensor<1x32x96x176x!qElemType>

    %1 = IE.Clamp(%0)
        {
            max = 6.000000e+00 : f64,
            min = 0.000000e+00 : f64
        } :
        tensor<1x32x96x176x!qElemType> -> tensor<1x32x96x176x!qElemType>

    return %1 : tensor<1x32x96x176x!qElemType>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x32x97x177x!qElemType> -> tensor<1x32x96x176x!qElemType>

    // CHECK-NOT:  IE.Clamp
}

// -----

// CHECK-LABEL: @FuseMaxPoolWithTanhTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithTanhTest(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %max_pool = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %tanh = IE.Tanh(%max_pool) : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %tanh : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Tanh<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Tanh
}

// -----

// CHECK-LABEL: @FuseMaxPoolWithBetaOneSwishTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithBetaOneSwishTest(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %max_pool = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %swish = IE.Swish(%max_pool) {beta_value = 1.000000e+00 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %swish : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Swish<beta = 1.000000e+00 : f64>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Swish
}

// CHECK-LABEL: @FuseMaxPoolWithBetaGreaterThanOneSwishTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithBetaGreaterThanOneSwishTest(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %max_pool = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %swish = IE.Swish(%max_pool) {beta_value = 1.700000e+00 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %swish : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Swish<beta = 1.700000e+00 : f64>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Swish
}

// -----

// CHECK-LABEL: @SwishWithLessThanOneBetaIsNotFused
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @SwishWithLessThanOneBetaIsNotFused(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %max_pool = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %swish = IE.Swish(%max_pool) {beta_value = 1.000000e-01 : f64} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %swish : tensor<1x16x3x3xf16>

    // CHECK-NOT: post_op
    // CHECK:     IE.Swish
}

// -----

// CHECK-LABEL: @FuseMaxPoolWithGeluTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithGeluTest(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %max_pool = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %tanh = IE.Gelu(%max_pool) : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %tanh : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Gelu<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Gelu
}

// -----

// CHECK-LABEL: @FuseConvWithQuantizedTanhTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x16x16xf16>)
func.func @FuseConvWithQuantizedTanhTest(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%input, %weights)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    %tanh = IE.Tanh(%conv) : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    %in_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %in_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %out_low = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %out_high = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %fake_quant = IE.FakeQuantize(%tanh, %in_low, %in_high, %out_low, %out_high)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } :
        tensor<1x16x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x16x16xf16>

    return %fake_quant : tensor<1x16x16x16xf16>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Tanh<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK-NOT:  IE.Tanh
}

// -----

// CHECK-LABEL: @DontFuseConvWithPerChannelQuantizedTanhTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x16x16xf16>)
func.func @DontFuseConvWithPerChannelQuantizedTanhTest(%input: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
    %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
    %conv = IE.Convolution(%input, %weights)
        {
            strides = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    %tanh = IE.Tanh(%conv) : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    %in_low_pt = const.Declare tensor<1x1x1x1xf16> = dense<-1.0> : tensor<1x1x1x1xf16>
    %in_high_pt = const.Declare tensor<1x1x1x1xf16> = dense<1.0> : tensor<1x1x1x1xf16>
    %out_low_pc = const.Declare tensor<1x16x1x1xf16> =
        dense<[[[[-0.7]],[[-0.5]],[[-0.3]],[[-0.7]],[[-0.6]],[[-0.9]],[[-0.4]],[[-0.7]],[[-0.5]],[[-0.3]],[[-0.7]],[[-0.5]],[[-0.3]],[[-0.7]],[[-0.5]],[[-0.3]]]]> : tensor<1x16x1x1xf16>
    %out_high_pc = const.Declare tensor<1x16x1x1xf16> =
        dense<[[[[0.7]],[[0.5]],[[0.3]],[[0.7]],[[0.6]],[[0.9]],[[0.4]],[[0.7]],[[0.5]],[[0.3]],[[0.7]],[[0.5]],[[0.3]],[[0.7]],[[0.5]],[[0.3]]]]> : tensor<1x16x1x1xf16>

    %fake_quant_pt = IE.FakeQuantize(%tanh, %in_low_pt, %in_high_pt, %in_high_pt, %in_high_pt)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } :
        tensor<1x16x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x16x16xf16>

    %fake_quant_pc = IE.FakeQuantize(%tanh, %in_low_pt, %in_high_pt, %out_low_pc, %out_high_pc)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } :
        tensor<1x16x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x16x16xf16>

    %add = IE.Add(%fake_quant_pt, %fake_quant_pc)
        {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>
        } :
        tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>

    return %add : tensor<1x16x16x16xf16>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[TANH:%.+]] = IE.Tanh([[CONV]]) : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>
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

// CHECK-LABEL: @NotFuseClampI32IntoAdd
// CHECK-SAME:    ([[INPUT1:%.+]]: tensor<1x50x1x1xsi32>, [[INPUT2:%.+]]: tensor<1xsi32>)

func.func @NotFuseClampI32IntoAdd(%arg0: tensor<1x50x1x1xsi32>, %arg1: tensor<1xsi32>) -> tensor<1x50x1x1xsi32> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x50x1x1xsi32>, tensor<1xsi32> -> tensor<1x50x1x1xsi32>
    %1 = IE.Clamp(%0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>
    return %1 : tensor<1x50x1x1xsi32>

    // CHECK:       [[Add:%.+]] = IE.Add([[INPUT1]], [[INPUT2]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x50x1x1xsi32>, tensor<1xsi32> -> tensor<1x50x1x1xsi32>
    // CHECK:       [[Clamp:%.+]] = IE.Clamp([[Add]]) {
    // CHECK-SAME:      max = 1.000000e+00 : f64,
    // CHECK-SAME:      min = -1.000000e+00 : f64
    // CHECK-SAME:  } : tensor<1x50x1x1xsi32> -> tensor<1x50x1x1xsi32>
}

// -----

// CHECK-LABEL: @FuseMaxPoolWithExpTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf16>)
func.func @FuseMaxPoolWithExpTest(%input: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Exp(%0) : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Exp<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    // CHECK-NOT:  IE.Exp
}

// -----

// CHECK-LABEL: @FailFP32FuseMaxPoolWithExpTest
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf32>)
func.func @FailFP32FuseMaxPoolWithExpTest(%input: tensor<1x16x4x4xf32>) -> tensor<1x16x3x3xf32> {
    %0 = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf32> -> tensor<1x16x3x3xf32>

    %1 = IE.Exp(%0) : tensor<1x16x3x3xf32> -> tensor<1x16x3x3xf32>

    return %1 : tensor<1x16x3x3xf32>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-NOT:      post_op = #IE.Exp<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf32> -> tensor<1x16x3x3xf32>

    // CHECK:  IE.Exp
}

// -----

// CHECK-LABEL: @FuseHSwishIntoMaxPool
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x4x4xf32>)
func.func @FuseHSwishIntoMaxPool(%input: tensor<1x16x4x4xf32>) -> tensor<1x16x3x3xf32> {
    %0 = IE.MaxPool(%input)
        {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>
        } :
        tensor<1x16x4x4xf32> -> tensor<1x16x3x3xf32>

    %1 = IE.HSwish(%0) : tensor<1x16x3x3xf32> -> tensor<1x16x3x3xf32>

    return %1 : tensor<1x16x3x3xf32>

    // CHECK:       [[MAX_POOL:%.+]] = IE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      kernel_size = [2, 2],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.HSwish<>,
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x4x4xf32> -> tensor<1x16x3x3xf32>

    // CHECK-NOT:  IE.HSwish
}

