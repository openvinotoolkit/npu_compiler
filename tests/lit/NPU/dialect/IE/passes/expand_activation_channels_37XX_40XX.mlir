//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandMaxPoolChannels
func.func @ExpandMaxPoolChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x13xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %0 : tensor<1x3x15x13xf16, {order = #NHWC}>
}

// CHECK:       [[PAD:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[POOL:%.*]] = IE.MaxPool([[PAD]])
// CHECK-SAME:      kernel_size = [5, 5]
// CHECK-SAME:      pads_begin = [2, 0]
// CHECK-SAME:      pads_end = [2, 0]
// CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
// CHECK-SAME:      strides = [2, 2]
// CHECK-SAME:      -> tensor<1x16x15x13xf16, {order = #NHWC}>

// CHECK:       [[OUT:%.*]] = IE.Slice [[POOL]] [0, 0, 0, 0] [1, 3, 15, 13]

// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandMaxPoolChannelsWithSliceProducer
func.func @ExpandMaxPoolChannelsWithSliceProducer(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x13xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 30, 30] : tensor<1x4x30x30xf16, {order = #NHWC}> to tensor<1x3x30x30xf16, {order = #NHWC}>

    %1 = IE.MaxPool(%0) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}> -> tensor<1x3x15x13xf16, {order = #NHWC}>

    return %1 : tensor<1x3x15x13xf16, {order = #NHWC}>
}

// CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 30, 30]
// CHECK:       [[EXPAND:%.+]] = IE.Expand([[SLICE]]) {pads_begin = [0, 1, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK:       [[POOL:%.*]] = IE.MaxPool([[EXPAND]])
// CHECK-SAME:      {kernel_size = [5, 5]
// CHECK-SAME:      pads_begin = [2, 0]
// CHECK-SAME:      pads_end = [2, 0]
// CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
// CHECK-SAME:      strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x13xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[POOL]] [0, 1, 0, 0] [1, 3, 15, 13]

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

// CHECK-LABEL: @NotExpandQuantMaxPoolChannels
func.func @NotExpandQuantMaxPoolChannels(%input: tensor<1x3x30x30x!qElemType, {order = #NHWC}>) -> tensor<1x3x15x13x!qElemType1, {order = #NHWC}> {
    %1 = IE.MaxPool(%input) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x3x15x13x!qElemType1, {order = #NHWC}>
    return %1 : tensor<1x3x15x13x!qElemType1, {order = #NHWC}>
}

// CHECK-NOT: IE.Expand
// CHECK-NOT: IE.Slice
// CHECK:     IE.MaxPool(%arg0) {kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x3x15x13x!qElemType1, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

// CHECK-LABEL: @ExpandQuantMaxPoolChannels
func.func @ExpandQuantMaxPoolChannels(%input: tensor<1x3x30x30x!qElemType, {order = #NHWC}>) -> tensor<1x3x15x13x!qElemType, {order = #NHWC}> {
    %1 = IE.MaxPool(%input) {
        kernel_size = [5, 5], pads_begin = [2, 0], pads_end = [2, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x3x15x13x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x3x15x13x!qElemType, {order = #NHWC}>
}

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK-SAME:  : tensor<1x3x30x30x!qElemType, {order = #NHWC}> -> tensor<1x16x30x30x!qElemType1, {order = #NHWC}>

// CHECK:       [[MAXPOOL_OUT:%.+]] = IE.MaxPool([[EXPAND_OUT]])
// CHECK-SAME:      kernel_size = [5, 5]
// CHECK-SAME:      pads_begin = [2, 0]
// CHECK-SAME:      pads_end = [2, 0]
// CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
// CHECK-SAME:      strides = [2, 2]
// CHECK-SAME:      -> tensor<1x16x15x13x!qElemType1, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[MAXPOOL_OUT]] [0, 0, 0, 0] [1, 3, 15, 13]
// CHECK-SAME:  : tensor<1x16x15x13x!qElemType1, {order = #NHWC}> to tensor<1x3x15x13x!qElemType, {order = #NHWC}>

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannels
func.func @ExpandEltwiseAddChannels(%arg0: tensor<1x3x30x25xf16, {order = #NHWC}>, %arg1: tensor<1x3x30x25xf16, {order = #NHWC}>)
        -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %0 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[EXPAND_LEFT_INPUT:%.*]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[EXPAND_RIGHT_INPUT:%.*]] = IE.Expand(%arg1) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[ELTWISE_ADD:%.*]] = IE.Add([[EXPAND_LEFT_INPUT]], [[EXPAND_RIGHT_INPUT]])
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_ADD]] [0, 0, 0, 0] [1, 3, 30, 25]
// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannelsWithSliceProducer
func.func @ExpandEltwiseAddChannelsWithSliceProducer(%arg0: tensor<1x4x30x25xf16, {order = #NHWC}>, %arg1: tensor<1x4x30x25xf16, {order = #NHWC}>)
        -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 30, 25] : tensor<1x4x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 1, 0, 0] [1, 3, 30, 25] : tensor<1x4x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>

    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[SLICE_LEFT:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 3, 30, 25]
// CHECK:       [[SLICE_RIGHT:%.+]] = IE.Slice %arg1 [0, 1, 0, 0] [1, 3, 30, 25]
// CHECK:       [[EXPAND_LEFT:%.+]] = IE.Expand([[SLICE_LEFT]]) {pads_begin = [0, 1, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK:       [[EXPAND_RIGHT:%.+]] = IE.Expand([[SLICE_RIGHT]]) {pads_begin = [0, 1, 0, 0], pads_end = [0, 12, 0, 0]}
// CHECK:       [[ELTWISE_ADD:%.*]] = IE.Add([[EXPAND_LEFT]], [[EXPAND_RIGHT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_ADD]] [0, 1, 0, 0] [1, 3, 30, 25]
// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannelsWithLeftOffsetSlice
func.func @ExpandEltwiseAddChannelsWithLeftOffsetSlice(%arg0: tensor<1x9x30x25xf16, {order = #NHWC}>, %arg1: tensor<1x10x30x25xf16, {order = #NHWC}>)
        -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 3, 0, 0] [1, 3, 30, 25] : tensor<1x9x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x10x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>

    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[SLICE_LEFT:%.+]] = IE.Slice %arg0 [0, 3, 0, 0] [1, 3, 30, 25] : tensor<1x9x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       [[SLICE_RIGHT:%.+]] = IE.Slice %arg1 [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x10x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       [[EXPAND_LEFT:%.+]] = IE.Expand([[SLICE_LEFT]]) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[EXPAND_RIGHT:%.+]] = IE.Expand([[SLICE_RIGHT]]) {pads_begin = [0, 3, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[ELTWISE_ADD:%.*]] = IE.Add([[EXPAND_LEFT]], [[EXPAND_RIGHT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x30x25xf16, {order = #NHWC}>, tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_ADD]] [0, 3, 0, 0] [1, 3, 30, 25] : tensor<1x16x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannelsWithRightOffsetSlice
func.func @ExpandEltwiseAddChannelsWithRightOffsetSlice(%arg0: tensor<1x9x30x25xf16, {order = #NHWC}>, %arg1: tensor<1x10x30x25xf16, {order = #NHWC}>)
        -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x9x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 3, 0, 0] [1, 3, 30, 25] : tensor<1x10x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>

    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %2 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK:       [[SLICE_LEFT:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x9x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       [[SLICE_RIGHT:%.+]] = IE.Slice %arg1 [0, 3, 0, 0] [1, 3, 30, 25] : tensor<1x10x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       [[EXPAND_LEFT:%.+]] = IE.Expand([[SLICE_LEFT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[EXPAND_RIGHT:%.+]] = IE.Expand([[SLICE_RIGHT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[ELTWISE_ADD:%.*]] = IE.Add([[EXPAND_LEFT]], [[EXPAND_RIGHT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x30x25xf16, {order = #NHWC}>, tensor<1x16x30x25xf16, {order = #NHWC}> -> tensor<1x16x30x25xf16, {order = #NHWC}>
// CHECK:       [[OUT:%.*]] = IE.Slice [[ELTWISE_ADD]] [0, 0, 0, 0] [1, 3, 30, 25] : tensor<1x16x30x25xf16, {order = #NHWC}> to tensor<1x3x30x25xf16, {order = #NHWC}>
// CHECK:       return [[OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvolutionChannels
func.func @ExpandGroupConvolutionChannels(%arg0: tensor<1x72x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<72x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x72x1x1xf16> = dense<1.0> : tensor<1x72x1x1xf16>

    %2 = IE.GroupConvolution(%arg0, %0, %1) {
        dilations = [1, 1], groups = 72, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x72x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_GROUP:%.*]] = const.Declare tensor<1x80x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 0, 0, 0]>]

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.GroupConvolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_GROUP]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]

// CHECK:       return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvolutionWithChannelsAndDilation
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x72x56x56xf16, {order = #NHWC}>
func.func @ExpandGroupConvolutionWithChannelsAndDilation(%arg0: tensor<1x72x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<72x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x72x1x1xf16> = dense<1.0> : tensor<1x72x1x1xf16>

    %2 = IE.GroupConvolution(%arg0, %0, %1) {
        dilations = [3, 3], groups = 72, pads_begin = [3, 3], pads_end = [3, 3], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %2 : tensor<1x72x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_GROUP:%.*]] = const.Declare tensor<1x80x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 0, 0, 0]>]

// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand([[ARG0]])

// CHECK:       [[EXTENDED_CONV:%.*]] = IE.GroupConvolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_GROUP]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]

// CHECK:       return [[REDUNDRANT_SUBTENSOR]]

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConvolutionChannelsWithSliceProducer
func.func @ExpandGroupConvolutionChannelsWithSliceProducer(%arg0: tensor<1x75x56x56xf16, {order = #NHWC}>) -> tensor<1x72x28x28xf16, {order = #NHWC}> {
    %0 = const.Declare tensor<72x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<72x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = const.Declare tensor<1x72x1x1xf16> = dense<1.0> : tensor<1x72x1x1xf16>

    %2 = IE.Slice %arg0 [0, 1, 0, 0] [1, 72, 56, 56] : tensor<1x75x56x56xf16, {order = #NHWC}> to tensor<1x72x56x56xf16, {order = #NHWC}>

    %3 = IE.GroupConvolution(%2, %0, %1) {
        dilations = [1, 1], groups = 72, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x72x56x56xf16, {order = #NHWC}>, tensor<72x1x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16> -> tensor<1x72x28x28xf16, {order = #NHWC}>

    return %3 : tensor<1x72x28x28xf16, {order = #NHWC}>
}

// CHECK-DAG:   [[EXTENDED_GROUP:%.*]] = const.Declare tensor<1x80x1x1xf16> =
// CHECK-SAME:      [#const.PadWithZero<[0, 1, 0, 0], [0, 7, 0, 0]>]
// CHECK-DAG:   [[EXTENDED_FILTER:%.*]] = const.Declare tensor<80x1x3x3xf16, {order = #NHWC}> =
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[1, 0, 0, 0], [7, 0, 0, 0]>]
// CHECK:       [[SLICE:%.+]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 72, 56, 56]
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand([[SLICE]]) {pads_begin = [0, 1, 0, 0], pads_end = [0, 7, 0, 0]}
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.GroupConvolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]], [[EXTENDED_GROUP]])
// CHECK:       [[SLICE_OUT:%.*]] = IE.Slice [[EXTENDED_CONV]] [0, 1, 0, 0] [1, 72, 28, 28]

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannels
func.func @ExpandConvolutionChannels(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<40x77x1x1xf16, {order = #NHWC}>) -> tensor<1x40x56x56xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<40x77x1x1xf16, {order = #NHWC}> -> tensor<1x40x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x40x56x56xf16, {order = #NHWC}>
}

// CHECK:       [[CST:%.*]] = const.Declare tensor<40x3x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<40x3x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[CONCAT:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<40x77x1x1xf16, {order = #NHWC}>, tensor<40x3x1x1xf16, {order = #NHWC}> -> tensor<40x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<40x80x1x1xf16, {order = #NHWC}> -> tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]
// CHECK:       return [[REDUNDRANT_SUBTENSOR]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844:127>

// CHECK-LABEL: @ExpandConvolutionChannelsWithQuant
func.func @ExpandConvolutionChannelsWithQuant(%arg0: tensor<1x77x56x56x!qElemType, {order = #NHWC}>, %arg1: tensor<40x77x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x40x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56x!qElemType, {order = #NHWC}>, tensor<40x77x1x1x!qElemType, {order = #NHWC}> -> tensor<1x40x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x40x56x56x!qElemType, {order = #NHWC}>
}

// CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<40x3x1x1x!qElemType, {order = #NHWC}> = dense<127> : tensor<40x3x1x1xui8, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_FILTER_IC:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<40x77x1x1x!qElemType, {order = #NHWC}>, tensor<40x3x1x1x!qElemType, {order = #NHWC}> -> tensor<40x80x1x1x!qElemType, {order = #NHWC}>
// CHECK:       [[EXTENDED_FILTER_OC:%.*]] = IE.Expand([[EXTENDED_FILTER_IC]]) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<40x80x1x1x!qElemType, {order = #NHWC}> -> tensor<48x80x1x1x!qElemType, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER_OC]])
// CHECK:       [[SLICE:%.+]] = IE.Slice [[EXTENDED_CONV]] [0, 0, 0, 0] [1, 40, 56, 56]
// CHECK:       return [[SLICE]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsOnlyIC
func.func @ExpandConvolutionChannelsOnlyIC(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<48x77x1x1xf16, {order = #NHWC}>) -> tensor<1x48x56x56xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<48x77x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x48x56x56xf16, {order = #NHWC}>
}

// CHECK:       [[CST:%.*]] = const.Declare tensor<48x3x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<48x3x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<48x77x1x1xf16, {order = #NHWC}>, tensor<48x3x1x1xf16, {order = #NHWC}> -> tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       return [[EXTENDED_CONV]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844:127>

// CHECK-LABEL: @ExpandConvolutionChannelsOnlyICWithQuant
func.func @ExpandConvolutionChannelsOnlyICWithQuant(%arg0: tensor<1x77x56x56x!qElemType, {order = #NHWC}>, %arg1: tensor<48x77x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x48x56x56x!qElemType, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56x!qElemType, {order = #NHWC}>, tensor<48x77x1x1x!qElemType, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x48x56x56x!qElemType, {order = #NHWC}>
}

// CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<48x3x1x1x!qElemType, {order = #NHWC}> = dense<127> : tensor<48x3x1x1xui8, {order = #NHWC}>
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_FILTER:%.*]] = IE.Concat(%arg1, [[CST]])
// CHECK{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 77, 0, 0]]} : tensor<48x77x1x1x!qElemType, {order = #NHWC}>, tensor<48x3x1x1x!qElemType, {order = #NHWC}> -> tensor<48x80x1x1x!qElemType, {order = #NHWC}>
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       return [[EXTENDED_CONV]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsConst
func.func @ExpandConvolutionChannelsConst(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>) -> tensor<1x40x56x56xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<40x77x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<40x77x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %filter)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x77x56x56xf16, {order = #NHWC}>, tensor<40x77x1x1xf16, {order = #NHWC}> -> tensor<1x40x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x40x56x56xf16, {order = #NHWC}>
}

// CHECK-DAG:       [[EXTENDED_FILTER:%.*]] = const.Declare tensor<48x80x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 3, 0, 0]>]
// CHECK:       [[EXTENDED_INPUT:%.*]] = IE.Expand(%arg0)
// CHECK:       [[EXTENDED_CONV:%.*]] = IE.Convolution([[EXTENDED_INPUT]], [[EXTENDED_FILTER]])
// CHECK:       [[REDUNDRANT_SUBTENSOR:%.*]] = IE.Slice [[EXTENDED_CONV]]
// CHECK:       return [[REDUNDRANT_SUBTENSOR]]


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

func.func @ExpandQuantGroupConvolutionChannels(
        %input: tensor<1x3x30x30x!qElemType, {order = #NHWC}>,
        %filter: tensor<3x1x3x3x!qElemType1, {order = #NHWC}>)
            -> tensor<1x3x15x15x!qElemType2, {order = #NHWC}> {
    %1 = IE.GroupConvolution(%input, %filter) {
        dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<3x1x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x3x15x15x!qElemType2, {order = #NHWC}>
    return %1 : tensor<1x3x15x15x!qElemType2, {order = #NHWC}>
}

// CHECK-LABEL: func.func @ExpandQuantGroupConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<3x1x3x3x!qElemType1, {order = #NHWC}>

// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[PAD_OUT:%.+]] = IE.Expand([[FILTER]]) {pads_begin = [0, 0, 0, 0], pads_end = [13, 0, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[PAD_OUT]])
// CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x15x!qElemType4, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15]

// CHECK:       return [[SLICE_OUT]] : tensor<1x3x15x15x!qElemType2, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ExpandQuantGroupConvolutionChannelsConst(%input: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x3x15x15xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<3x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x1x3x3xf16>, [#const.Reorder<#NHWC>]
    %1 = IE.GroupConvolution(%input, %filter) {
        dilations = [1, 1], groups = 3, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<3x1x3x3xf16, {order = #NHWC}> -> tensor<1x3x15x15xf16, {order = #NHWC}>
    return %1 : tensor<1x3x15x15xf16, {order = #NHWC}>
}

// CHECK-LABEL: func.func @ExpandQuantGroupConvolutionChannelsConst
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30xf16, {order = #NHWC}>

// CHECK-DAG:       [[EXPAND_FILTER:%.*]] = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>]
// CHECK:       [[EXPAND_OUT:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}

// CHECK:       [[CONV_OUT:%.+]] = IE.GroupConvolution([[EXPAND_OUT]], [[EXPAND_FILTER]])
// CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
// CHECK-SAME:      -> tensor<1x16x15x15xf16, {order = #NHWC}>

// CHECK:       [[SLICE_OUT:%.+]] = IE.Slice [[CONV_OUT]] [0, 0, 0, 0] [1, 3, 15, 15]

// CHECK:       return [[SLICE_OUT]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PreventConcat
func.func @PreventConcat(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<48x77x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<1x48x56x56xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [48, 38, 1, 1] : tensor<48x77x1x1xf16, {order = #NHWC}> to tensor<48x38x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 38, 0, 0] [48, 39, 1, 1] : tensor<48x77x1x1xf16, {order = #NHWC}> to tensor<48x39x1x1xf16, {order = #NHWC}>

    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x38x56x56xf16, {order = #NHWC}>
    %3 = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x39x56x56xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %0)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x38x56x56xf16, {order = #NHWC}>, tensor<48x38x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

    %5 = IE.Convolution(%3, %1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x39x56x56xf16, {order = #NHWC}>, tensor<48x39x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

    return %4, %5 : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<1x48x56x56xf16, {order = #NHWC}>
}

// CHECK: [[CST:%.*]] = const.Declare tensor<48x9x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<48x9x1x1xf16, {order = #NHWC}>
// CHECK: [[CST_0:%.*]] = const.Declare tensor<48x10x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<48x10x1x1xf16, {order = #NHWC}>
// CHECK: [[SLICE0:%.+]] = IE.Slice %arg1 [0, 0, 0, 0] [48, 38, 1, 1] : tensor<48x77x1x1xf16, {order = #NHWC}> to tensor<48x38x1x1xf16, {order = #NHWC}>
// CHECK: [[SLICE1:%.+]] = IE.Slice %arg1 [0, 38, 0, 0] [48, 39, 1, 1] : tensor<48x77x1x1xf16, {order = #NHWC}> to tensor<48x39x1x1xf16, {order = #NHWC}>

// CHECK: [[SLICE2:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x38x56x56xf16, {order = #NHWC}>
// CHECK: [[SLICE3:%.+]] = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x39x56x56xf16, {order = #NHWC}>

// CHECK: [[EXPAND0:%.+]] = IE.Expand([[SLICE2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x38x56x56xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 38, 0, 0]]} : tensor<48x38x1x1xf16, {order = #NHWC}>, tensor<48x10x1x1xf16, {order = #NHWC}> -> tensor<48x48x1x1xf16, {order = #NHWC}>

// CHECK: [[CONVOLUTION0:%.+]] = IE.Convolution([[EXPAND0]], [[CONCAT0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

// CHECK: [[EXPAND2:%.+]] = IE.Expand([[SLICE3]]) {pads_begin = [0, 9, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x39x56x56xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[CONCAT2:%.+]] = IE.Concat([[CST]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 9, 0, 0]]} : tensor<48x9x1x1xf16, {order = #NHWC}>, tensor<48x39x1x1xf16, {order = #NHWC}> -> tensor<48x48x1x1xf16, {order = #NHWC}>

// CHECK: [[CONVOLUTION1:%.+]] = IE.Convolution([[EXPAND2]], [[CONCAT2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>

// CHECK: return [[CONVOLUTION0]], [[CONVOLUTION1]] : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<1x48x56x56xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PreventConcat2DimsExpand
func.func @PreventConcat2DimsExpand(%arg0: tensor<1x77x56x56xf16, {order = #NHWC}>, %arg1: tensor<46x77x1x1xf16, {order = #NHWC}>)
    -> (tensor<1x46x56x56xf16, {order = #NHWC}>, tensor<1x46x56x56xf16, {order = #NHWC}>) {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [46, 38, 1, 1] : tensor<46x77x1x1xf16, {order = #NHWC}> to tensor<46x38x1x1xf16, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 38, 0, 0] [46, 39, 1, 1] : tensor<46x77x1x1xf16, {order = #NHWC}> to tensor<46x39x1x1xf16, {order = #NHWC}>

    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x38x56x56xf16, {order = #NHWC}>
    %3 = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x39x56x56xf16, {order = #NHWC}>

    %4 = IE.Convolution(%2, %0)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x38x56x56xf16, {order = #NHWC}>, tensor<46x38x1x1xf16, {order = #NHWC}> -> tensor<1x46x56x56xf16, {order = #NHWC}>

    %5 = IE.Convolution(%3, %1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x39x56x56xf16, {order = #NHWC}>, tensor<46x39x1x1xf16, {order = #NHWC}> -> tensor<1x46x56x56xf16, {order = #NHWC}>

    return %4, %5 : tensor<1x46x56x56xf16, {order = #NHWC}>, tensor<1x46x56x56xf16, {order = #NHWC}>
}

// CHECK: [[CST:%.*]] = const.Declare tensor<46x9x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<46x9x1x1xf16, {order = #NHWC}>
// CHECK: [[CST_0:%.*]] = const.Declare tensor<46x10x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<46x10x1x1xf16, {order = #NHWC}>
// CHECK: [[SLICE0:%.+]] = IE.Slice %arg1 [0, 0, 0, 0] [46, 38, 1, 1] : tensor<46x77x1x1xf16, {order = #NHWC}> to tensor<46x38x1x1xf16, {order = #NHWC}>
// CHECK: [[SLICE1:%.+]] = IE.Slice %arg1 [0, 38, 0, 0] [46, 39, 1, 1] : tensor<46x77x1x1xf16, {order = #NHWC}> to tensor<46x39x1x1xf16, {order = #NHWC}>

// CHECK: [[SLICE2:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x38x56x56xf16, {order = #NHWC}>
// CHECK: [[SLICE3:%.+]] = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56xf16, {order = #NHWC}> to tensor<1x39x56x56xf16, {order = #NHWC}>

// CHECK: [[EXPAND0:%.+]] = IE.Expand([[SLICE2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x38x56x56xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[CONCAT1:%.+]] = IE.Concat([[SLICE0]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 38, 0, 0]]} : tensor<46x38x1x1xf16, {order = #NHWC}>, tensor<46x10x1x1xf16, {order = #NHWC}> -> tensor<46x48x1x1xf16, {order = #NHWC}>
// CHECK: [[EXPAND2:%.+]] = IE.Expand([[CONCAT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [2, 0, 0, 0]} : tensor<46x48x1x1xf16, {order = #NHWC}> -> tensor<48x48x1x1xf16, {order = #NHWC}>

// CHECK: [[CONVOLUTION0:%.+]] = IE.Convolution([[EXPAND0]], [[EXPAND2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[RESULT0:%.+]] = IE.Slice [[CONVOLUTION0]] [0, 0, 0, 0] [1, 46, 56, 56] : tensor<1x48x56x56xf16, {order = #NHWC}> to tensor<1x46x56x56xf16, {order = #NHWC}>

// CHECK: [[EXPAND3:%.+]] = IE.Expand([[SLICE3]]) {pads_begin = [0, 9, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x39x56x56xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[CONCAT4:%.+]] = IE.Concat([[CST]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 9, 0, 0]]} : tensor<46x9x1x1xf16, {order = #NHWC}>, tensor<46x39x1x1xf16, {order = #NHWC}> -> tensor<46x48x1x1xf16, {order = #NHWC}>
// CHECK: [[EXPAND5:%.+]] = IE.Expand([[CONCAT4]]) {pads_begin = [0, 0, 0, 0], pads_end = [2, 0, 0, 0]} : tensor<46x48x1x1xf16, {order = #NHWC}> -> tensor<48x48x1x1xf16, {order = #NHWC}>

// CHECK: [[CONVOLUTION1:%.+]] = IE.Convolution([[EXPAND3]], [[EXPAND5]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56xf16, {order = #NHWC}>, tensor<48x48x1x1xf16, {order = #NHWC}> -> tensor<1x48x56x56xf16, {order = #NHWC}>
// CHECK: [[RESULT1:%.+]] = IE.Slice [[CONVOLUTION1]] [0, 0, 0, 0] [1, 46, 56, 56] : tensor<1x48x56x56xf16, {order = #NHWC}> to tensor<1x46x56x56xf16, {order = #NHWC}>

// CHECK: return [[RESULT0]], [[RESULT1]] : tensor<1x46x56x56xf16, {order = #NHWC}>, tensor<1x46x56x56xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844:127>

// CHECK-LABEL: @U8Type2DimsExpand
func.func @U8Type2DimsExpand(%arg0: tensor<1x77x56x56x!qElemType, {order = #NHWC}>, %arg1: tensor<46x77x1x1x!qElemType, {order = #NHWC}>)
    -> (tensor<1x46x56x56x!qElemType, {order = #NHWC}>, tensor<1x46x56x56x!qElemType, {order = #NHWC}>) {
    %0 = IE.Slice %arg1 [0, 0, 0, 0] [46, 38, 1, 1] : tensor<46x77x1x1x!qElemType, {order = #NHWC}> to tensor<46x38x1x1x!qElemType, {order = #NHWC}>
    %1 = IE.Slice %arg1 [0, 38, 0, 0] [46, 39, 1, 1] : tensor<46x77x1x1x!qElemType, {order = #NHWC}> to tensor<46x39x1x1x!qElemType, {order = #NHWC}>

    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56x!qElemType, {order = #NHWC}> to tensor<1x38x56x56x!qElemType, {order = #NHWC}>
    %3 = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56x!qElemType, {order = #NHWC}> to tensor<1x39x56x56x!qElemType, {order = #NHWC}>

    %4 = IE.Convolution(%2, %0)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x38x56x56x!qElemType, {order = #NHWC}>, tensor<46x38x1x1x!qElemType, {order = #NHWC}> -> tensor<1x46x56x56x!qElemType, {order = #NHWC}>

    %5 = IE.Convolution(%3, %1)
        {
            strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
        } : tensor<1x39x56x56x!qElemType, {order = #NHWC}>, tensor<46x39x1x1x!qElemType, {order = #NHWC}> -> tensor<1x46x56x56x!qElemType, {order = #NHWC}>

    return %4, %5 : tensor<1x46x56x56x!qElemType, {order = #NHWC}>, tensor<1x46x56x56x!qElemType, {order = #NHWC}>
}

// CHECK-DAG: [[CONST0:%.*]] = const.Declare tensor<46x9x1x1x!qElemType, {order = #NHWC}> = dense<127> : tensor<46x9x1x1xui8, {order = #NHWC}>
// CHECK-DAG: [[CONST1:%.*]] = const.Declare tensor<46x10x1x1x!qElemType, {order = #NHWC}> = dense<127> : tensor<46x10x1x1xui8, {order = #NHWC}>

// CHECK: [[SLICE0:%.+]] = IE.Slice %arg1 [0, 0, 0, 0] [46, 38, 1, 1] : tensor<46x77x1x1x!qElemType, {order = #NHWC}> to tensor<46x38x1x1x!qElemType, {order = #NHWC}>
// CHECK: [[SLICE1:%.+]] = IE.Slice %arg1 [0, 38, 0, 0] [46, 39, 1, 1] : tensor<46x77x1x1x!qElemType, {order = #NHWC}> to tensor<46x39x1x1x!qElemType, {order = #NHWC}>

// CHECK: [[SLICE2:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 38, 56, 56] : tensor<1x77x56x56x!qElemType, {order = #NHWC}> to tensor<1x38x56x56x!qElemType, {order = #NHWC}>
// CHECK: [[SLICE3:%.+]] = IE.Slice %arg0 [0, 38, 0, 0] [1, 39, 56, 56] : tensor<1x77x56x56x!qElemType, {order = #NHWC}> to tensor<1x39x56x56x!qElemType, {order = #NHWC}>

// CHECK: [[EXPAND0:%.+]] = IE.Expand([[SLICE2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x38x56x56x!qElemType, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType, {order = #NHWC}>
// CHECK: [[CONCAT0:%.*]] = IE.Concat([[SLICE0]], [[CONST1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 38, 0, 0]]} : tensor<46x38x1x1x!qElemType, {order = #NHWC}>, tensor<46x10x1x1x!qElemType, {order = #NHWC}> -> tensor<46x48x1x1x!qElemType, {order = #NHWC}>

// CHECK: [[EXPAND1:%.+]] = IE.Expand([[CONCAT0]]) {pads_begin = [0, 0, 0, 0], pads_end = [2, 0, 0, 0]} : tensor<46x48x1x1x!qElemType, {order = #NHWC}> -> tensor<48x48x1x1x!qElemType, {order = #NHWC}>

// CHECK: [[CONVOLUTION0:%.+]] = IE.Convolution([[EXPAND0]], [[EXPAND1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56x!qElemType, {order = #NHWC}>, tensor<48x48x1x1x!qElemType, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType, {order = #NHWC}>
// CHECK: [[RESULT0:%.+]] = IE.Slice [[CONVOLUTION0]] [0, 0, 0, 0] [1, 46, 56, 56] : tensor<1x48x56x56x!qElemType, {order = #NHWC}> to tensor<1x46x56x56x!qElemType, {order = #NHWC}>

// CHECK: [[EXPAND2:%.+]] = IE.Expand([[SLICE3]]) {pads_begin = [0, 9, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x39x56x56x!qElemType, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType, {order = #NHWC}>
// CHECK: [[CONCAT1:%.*]] = IE.Concat([[CONST0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 9, 0, 0]]} : tensor<46x9x1x1x!qElemType, {order = #NHWC}>, tensor<46x39x1x1x!qElemType, {order = #NHWC}> -> tensor<46x48x1x1x!qElemType, {order = #NHWC}>
// CHECK: [[EXPAND3:%.+]] = IE.Expand([[CONCAT1]]) {pads_begin = [0, 0, 0, 0], pads_end = [2, 0, 0, 0]} : tensor<46x48x1x1x!qElemType, {order = #NHWC}> -> tensor<48x48x1x1x!qElemType, {order = #NHWC}>

// CHECK: [[CONVOLUTION1:%.+]] = IE.Convolution([[EXPAND2]], [[EXPAND3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x56x56x!qElemType, {order = #NHWC}>, tensor<48x48x1x1x!qElemType, {order = #NHWC}> -> tensor<1x48x56x56x!qElemType, {order = #NHWC}>
// CHECK: [[RESULT1:%.+]] = IE.Slice [[CONVOLUTION1]] [0, 0, 0, 0] [1, 46, 56, 56] : tensor<1x48x56x56x!qElemType, {order = #NHWC}> to tensor<1x46x56x56x!qElemType, {order = #NHWC}>

// CHECK: return [[RESULT0]], [[RESULT1]] : tensor<1x46x56x56x!qElemType, {order = #NHWC}>, tensor<1x46x56x56x!qElemType, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotExpandInterpolateNearestChannels
func.func @DoNotExpandInterpolateNearestChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x3x30x30xf16> -> tensor<1x3x60x60xf16>

    return %0 : tensor<1x3x60x60xf16>
}

// CHECK-NOT:   IE.Expand

// CHECK:       [[INTERP:%.+]] = IE.Interpolate(%arg0)
// CHECK-SAME:      attr = #IE.Interpolate<mode = <NEAREST>,
// CHECK-SAME:                             shape_calc_mode = <SCALES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [2, 3],
// CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
// CHECK-SAME:      sizes_attr = [60, 60]
// CHECK-SAME:      -> tensor<1x3x60x60xf16>

// CHECK-NOT:   IE.Slice

// CHECK:       return [[INTERP]]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotExpandInterpolateLinearChannels
func.func @DoNotExpandInterpolateLinearChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x60x60xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [60, 60]
         } : tensor<1x3x30x30xf16> -> tensor<1x3x60x60xf16>

    return %0 : tensor<1x3x60x60xf16>
}

// CHECK-NOT:   IE.Expand

// CHECK:       [[INTERP:%.+]] = IE.Interpolate(%arg0)
// CHECK-SAME:      attr = #IE.Interpolate<mode = <LINEAR>,
// CHECK-SAME:                             shape_calc_mode = <SCALES>,
// CHECK-SAME:                             coord_mode = <ASYMMETRIC>,
// CHECK-SAME:                             nearest_mode = <FLOOR>,
// CHECK-SAME:                             antialias = false,
// CHECK-SAME:                             pads_begin = [0, 0, 0, 0],
// CHECK-SAME:                             pads_end = [0, 0, 0, 0],
// CHECK-SAME:                             cube_coeff = -7.500000e-01 : f64>,
// CHECK-SAME:      axes_attr = [2, 3],
// CHECK-SAME:      scales_attr = [2.000000e+00, 2.000000e+00],
// CHECK-SAME:      sizes_attr = [60, 60]
// CHECK-SAME:      -> tensor<1x3x60x60xf16>

// CHECK-NOT:   IE.Slice

// CHECK:       return [[INTERP]]

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @main {

    config.PipelineOptions @Options {
            config.Option @config.FP16CompressedConv : true
    }

    // CHECK-LABEL: @ConvToNCE4channels
    // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x3x16x16xf16, {order = #NHWC}>)
    func.func @ConvToNCE4channels(%arg0: tensor<1x3x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<16x3x1x1xf16, {order = #NHWC}> =
            dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>]
        %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

        %0 = IE.Convolution(%arg0, %weights, %bias) {
                dilations = [1, 1],
                pads_begin = [0, 0],
                pads_end = [0, 0],
                strides = [1, 1],
                post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
            } : tensor<1x3x16x16xf16, {order = #NHWC}>, tensor<16x3x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
                -> tensor<1x16x16x16xf16, {order = #NHWC}>

        return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>,
        // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
        // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<1x16x1x1xf16>

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
        // CHECK-SAME:      : tensor<1x3x16x16xf16, {order = #NHWC}> -> tensor<1x4x16x16xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
        // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]}
        // CHECK-SAME:      : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
        // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

        // CHECK:       return [[CONV]] : tensor<1x16x16x16xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsWithSoftMaxAfter
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x512x56x56xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<510x512x1x1xf16, {order = #NHWC}>)
func.func @ExpandConvolutionChannelsWithSoftMaxAfter(%arg0: tensor<1x512x56x56xf16, {order = #NHWC}>, %arg1: tensor<510x512x1x1xf16, {order = #NHWC}>) -> tensor<1x510x56x56xf16, {order = #NHWC}> {

    %0 = IE.Convolution(%arg0, %arg1) {
        strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1]
    } : tensor<1x512x56x56xf16, {order = #NHWC}>, tensor<510x512x1x1xf16, {order = #NHWC}> -> tensor<1x510x56x56xf16, {order = #NHWC}>
    %1 = IE.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x510x56x56xf16, {order = #NHWC}> -> tensor<1x510x56x56xf16, {order = #NHWC}>

    return %1 : tensor<1x510x56x56xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[WEIGHTS]]) {pads_begin = [0, 0, 0, 0], pads_end = [2, 0, 0, 0]} : tensor<510x512x1x1xf16, {order = #NHWC}> -> tensor<512x512x1x1xf16, {order = #NHWC}>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x512x56x56xf16, {order = #NHWC}>

    // CHECK:    [[SOFTMAX:%.+]] = IE.SoftMax([[CONV]]) 
    // CHECK-SAME:        axisInd = 1 : i64
    // CHECK-SAME:        padSize = 2 : i64
    // CHECK-SAME:        -> tensor<1x512x56x56xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[SOFTMAX]]
    // CHECK-SAME:        tensor<1x510x56x56xf16, {order = #NHWC}>

    // CHECK:    return [[SLICE]] : tensor<1x510x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsDynamic
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>)
func.func @ExpandConvolutionChannelsDynamic(%arg0: tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}> {

    %cst = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]
    } : tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    return %0 : tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 720, 1280]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:        -> tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:    return [[CONV]] : tensor<1x32x360x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 360, 640]> : tensor<4xsi64>, order = #NHWC}>
}
