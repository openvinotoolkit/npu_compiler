//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% num-of-dpu-groups=1 enable-auto-padding-idu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ZMajorConvExpandInChAutopadFP16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @ZMajorConvExpandInChAutopadFP16(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ZMajorConvExpandInChGreaterAutopadFP16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x30x30xf16, {order = #NHWC}>)
func.func @ZMajorConvExpandInChGreaterAutopadFP16(%arg0: tensor<1x12x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x12x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x12x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x12x30x30xf16, {order = #NHWC}>, tensor<5x12x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 4, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BiasConvolutionAutopadInChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @BiasConvolutionAutopadInChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x5x1x1xf16> = dense<1.0> : tensor<1x5x1x1xf16>

    %2 = IE.Convolution(%arg0, %0, %1) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}>, tensor<1x5x1x1xf16> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST_WEIGHTS]], [[CST_BIAS]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandZMajorConvChannelsOnlyWeightFP16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x30x30xf16, {order = #NHWC}>)
func.func @ExpandZMajorConvChannelsOnlyWeightFP16(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x4x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<5x4x3x3xf16>, [#const.Reorder<#NHWC>]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x4x30x30xf16, {order = #NHWC}>, tensor<5x4x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 12, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK:  !qElemType = !quant.uniform<i4:f16, 1.1534313725490195>

// CHECK-LABEL: @ExpandZMajorConvChannelsMixedPrecisionI4Quant
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @ExpandZMajorConvChannelsMixedPrecisionI4Quant(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3x!qElemType, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [
            #const.Reorder<#NHWC>,
            #const.CastElemType<si4>,
            #const.CastElemType<!qElemType>
    ]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>

    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3x!qElemType, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvolutionAutopadInChannelsDynamic
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ConvolutionAutopadInChannelsDynamic(%arg0: tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 0, 0, 0]
    // CHECK-SAME:        -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>
}
