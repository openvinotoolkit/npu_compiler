//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-sprlut=true" --fuse-activation-ops %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @SwapSliceWithSwishAndFuseIntoGroupConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x6144x516x1xf16>)
func.func @SwapSliceWithSwishAndFuseIntoGroupConv(%arg0: tensor<1x6144x516x1xf16>) -> tensor<1x6144x512x1xf16> {
    %weights = const.Declare tensor<6144x1x4x1xf16> = dense<1.0> : tensor<6144x1x4x1xf16>
    %conv = IE.GroupConvolution(%arg0, %weights)
        {
            dilations = [1, 1],
            groups = 6144 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x6144x516x1xf16>, tensor<6144x1x4x1xf16> -> tensor<1x6144x513x1xf16>

    %slice = IE.Slice %conv [0, 0, 1, 0] [1, 6144, 512, 1] : tensor<1x6144x513x1xf16> to tensor<1x6144x512x1xf16>
    %swish = IE.Swish(%slice) {beta_value = 1.000000e+00 : f64} : tensor<1x6144x512x1xf16> -> tensor<1x6144x512x1xf16>

    return %swish : tensor<1x6144x512x1xf16>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<6144x1x4x1xf16> = dense<1.000000e+00> : tensor<6144x1x4x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 6144 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Swish<beta = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x6144x516x1xf16>, tensor<6144x1x4x1xf16> -> tensor<1x6144x513x1xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 1, 0] [1, 6144, 512, 1]
    // CHECK-NOT:   IE.Swish
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @SwapSliceWithSwishAndFuseIntoConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x18x18xf16>)
func.func @SwapSliceWithSwishAndFuseIntoConv(%arg0: tensor<1x16x18x18xf16>) -> tensor<1x16x15x15xf16> {
    %weights = const.Declare tensor<16x16x3x3xf16> = dense<1.0> : tensor<16x16x3x3xf16>
    %conv = IE.Convolution(%arg0, %weights)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x16x18x18xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x16x16xf16>

    %slice = IE.Slice %conv [0, 0, 1, 1] [1, 16, 15, 15] : tensor<1x16x16x16xf16> to tensor<1x16x15x15xf16>
    %swish = IE.Swish(%slice) {beta_value = 1.000000e+00 : f64} : tensor<1x16x15x15xf16> -> tensor<1x16x15x15xf16>

    return %swish : tensor<1x16x15x15xf16>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Swish<beta = 1.000000e+00 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x18x18xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x16x16xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 1, 1] [1, 16, 15, 15]
    // CHECK-NOT:   IE.Swish
    // CHECK:       return [[SLICE]]
}

// -----

// CHECK-LABEL: @NoSwapWhenSliceHasMultipleUsers
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x18x18xf16>)
func.func @NoSwapWhenSliceHasMultipleUsers(%arg0: tensor<1x16x18x18xf16>) -> (tensor<1x16x15x15xf16>, tensor<1x16x15x15xf16>) {
    %weights = const.Declare tensor<16x16x3x3xf16> = dense<1.0> : tensor<16x16x3x3xf16>
    %conv = IE.Convolution(%arg0, %weights)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x16x18x18xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x16x16xf16>

    %slice = IE.Slice %conv [0, 0, 1, 1] [1, 16, 15, 15] : tensor<1x16x16x16xf16> to tensor<1x16x15x15xf16>
    %swish = IE.Swish(%slice) {beta_value = 1.000000e+00 : f64} : tensor<1x16x15x15xf16> -> tensor<1x16x15x15xf16>

    return %swish, %slice : tensor<1x16x15x15xf16>, tensor<1x16x15x15xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]]
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[SLICE]])
    // CHECK:       return [[SWISH]], [[SLICE]]
}

// -----

// CHECK-LABEL: @NoSwapWhenProducerHasMultipleUsers
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x6144x516x1xf16>)
func.func @NoSwapWhenProducerHasMultipleUsers(%arg0: tensor<1x6144x516x1xf16>) -> (tensor<1x6144x512x1xf16>, tensor<1x6144x256x1xf16>) {
    %weights = const.Declare tensor<6144x1x4x1xf16> = dense<1.0> : tensor<6144x1x4x1xf16>
    %conv = IE.GroupConvolution(%arg0, %weights)
        {
            dilations = [1, 1],
            groups = 6144 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x6144x516x1xf16>, tensor<6144x1x4x1xf16> -> tensor<1x6144x513x1xf16>

    %slice1 = IE.Slice %conv [0, 0, 1, 0] [1, 6144, 512, 1] : tensor<1x6144x513x1xf16> to tensor<1x6144x512x1xf16>
    %swish = IE.Swish(%slice1) {beta_value = 1.000000e+00 : f64} : tensor<1x6144x512x1xf16> -> tensor<1x6144x512x1xf16>
    %slice2 = IE.Slice %conv [0, 0, 0, 0] [1, 6144, 256, 1] : tensor<1x6144x513x1xf16> to tensor<1x6144x256x1xf16>

    return %swish, %slice2 : tensor<1x6144x512x1xf16>, tensor<1x6144x256x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution
    // CHECK-NOT:   post_op
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONV]] [0, 0, 1, 0]
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[SLICE1]])
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0]
    // CHECK:       return [[SWISH]], [[SLICE2]]
}

// -----

// CHECK-LABEL: @NoSwapWhenSwishBetaLessThanOne
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x6144x516x1xf16>)
func.func @NoSwapWhenSwishBetaLessThanOne(%arg0: tensor<1x6144x516x1xf16>) -> tensor<1x6144x512x1xf16> {
    %weights = const.Declare tensor<6144x1x4x1xf16> = dense<1.0> : tensor<6144x1x4x1xf16>
    %conv = IE.GroupConvolution(%arg0, %weights)
        {
            dilations = [1, 1],
            groups = 6144 : i64,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } :
        tensor<1x6144x516x1xf16>, tensor<6144x1x4x1xf16> -> tensor<1x6144x513x1xf16>

    %slice = IE.Slice %conv [0, 0, 1, 0] [1, 6144, 512, 1] : tensor<1x6144x513x1xf16> to tensor<1x6144x512x1xf16>
    %swish = IE.Swish(%slice) {beta_value = 5.000000e-01 : f64} : tensor<1x6144x512x1xf16> -> tensor<1x6144x512x1xf16>

    return %swish : tensor<1x6144x512x1xf16>

    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution
    // CHECK-NOT:   post_op
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]]
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[SLICE]])
    // CHECK:       return [[SWISH]]
}

// -----

// CHECK-LABEL: @NoSwapWhenProducerAlreadyHasPostOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x16x18x18xf16>)
func.func @NoSwapWhenProducerAlreadyHasPostOp(%arg0: tensor<1x16x18x18xf16>) -> tensor<1x16x15x15xf16> {
    %weights = const.Declare tensor<16x16x3x3xf16> = dense<1.0> : tensor<16x16x3x3xf16>
    %conv = IE.Convolution(%arg0, %weights)
        {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            post_op = #IE.Relu<>,
            strides = [1, 1]
        } :
        tensor<1x16x18x18xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x16x16xf16>

    %slice = IE.Slice %conv [0, 0, 1, 1] [1, 16, 15, 15] : tensor<1x16x16x16xf16> to tensor<1x16x15x15xf16>
    %swish = IE.Swish(%slice) {beta_value = 1.000000e+00 : f64} : tensor<1x16x15x15xf16> -> tensor<1x16x15x15xf16>

    return %swish : tensor<1x16x15x15xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK-SAME:      post_op = #IE.Relu<>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]]
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[SLICE]])
    // CHECK:       return [[SWISH]]
}
