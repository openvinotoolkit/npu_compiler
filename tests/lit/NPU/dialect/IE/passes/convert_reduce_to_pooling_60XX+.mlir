//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-is-reduce-supported=true" --convert-reduce-to-pooling="enable-fuse-reduce-minmax-to-dpu=true" %s | FileCheck %s
// REQUIRES: platform-NPU6010

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotConvertReduceMaxOnChannelAxisWithDPUParentSameLayout
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16, {order = #NHWC}>)
func.func @DoNotConvertReduceMaxOnChannelAxisWithDPUParentSameLayout(%arg0: tensor<1x8x8x8xf16, {order = #NHWC}>) -> tensor<1x1x8x8xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    %0 = IE.ReduceMax(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16, {order = #NHWC}>
    return %0 : tensor<1x1x8x8xf16, {order = #NHWC}>

    // Parent DPU op does not permute its output (input and output share NHWC layout),
    // so the channel-axis fusion opportunity is preserved and the reduce is not converted.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK-NOT: IE.MaxPool
    // CHECK:     IE.ReduceMax([[CONV]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotConvertReduceMinOnChannelAxisWithDPUParentSameLayout
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16, {order = #NHWC}>)
func.func @DoNotConvertReduceMinOnChannelAxisWithDPUParentSameLayout(%arg0: tensor<1x8x8x8xf16, {order = #NHWC}>) -> tensor<1x1x8x8xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16, {order = #NHWC}>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    %0 = IE.ReduceMin(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16, {order = #NHWC}>
    return %0 : tensor<1x1x8x8xf16, {order = #NHWC}>

    // Parent DPU op does not permute its output (input and output share NHWC layout),
    // so the channel-axis fusion opportunity is preserved and the reduce is not converted.
    // CHECK:     [[CONV:%.+]] = IE.Convolution
    // CHECK-NOT: IE.Negative
    // CHECK:     IE.ReduceMin([[CONV]])
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReduceMaxOnChannelAxisWithDPUParentPermutingOutput
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16>)
func.func @ConvertReduceMaxOnChannelAxisWithDPUParentPermutingOutput(%arg0: tensor<1x8x8x8xf16>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    %0 = IE.ReduceMax(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // Parent DPU op produces NHWC output from NCHW input, permuting the layout.
    // The channel-axis fusion opportunity does not apply, so the reduce is converted to pooling.
    // CHECK:     IE.Convolution
    // CHECK-NOT: IE.ReduceMax
    // CHECK:     IE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertReduceMinOnChannelAxisWithDPUParentPermutingOutput
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x8x8x8xf16>)
func.func @ConvertReduceMinOnChannelAxisWithDPUParentPermutingOutput(%arg0: tensor<1x8x8x8xf16>) -> tensor<1x1x8x8xf16> {
    %weights = const.Declare tensor<8x8x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x8x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
                : tensor<1x8x8x8xf16>, tensor<8x8x1x1xf16, {order = #NHWC}> -> tensor<1x8x8x8xf16, {order = #NHWC}>
    %0 = IE.ReduceMin(%conv) {axes_value = [1], keep_dims} : tensor<1x8x8x8xf16, {order = #NHWC}> -> tensor<1x1x8x8xf16>
    return %0 : tensor<1x1x8x8xf16>

    // Parent DPU op produces NHWC output from NCHW input, permuting the layout.
    // The channel-axis fusion opportunity does not apply, so the reduce is converted to pooling.
    // CHECK:     IE.Convolution
    // CHECK-NOT: IE.ReduceMin
    // CHECK:     IE.Negative
}
