//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% num-of-dpu-groups=1 enable-auto-padding-odu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ZMajorConvExpandInChAutopadOutChFP16
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @ZMajorConvExpandInChAutopadOutChFP16(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
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

// CHECK-LABEL: @ExpandBiasesConvolutionAutopadOutChannels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @ExpandBiasesConvolutionAutopadOutChannels(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
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
!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<i4:f16, 0.0173492431640625>
!qElemType2 = !quant.uniform<i4:f16, 0.012699142156862745>
// CHECK:  !qElemType = !quant.uniform<i4:f16, 0.0173492431640625>
// CHECK:  !qElemType1 = !quant.uniform<i4:f16, 0.012699142156862745>
// CHECK:  !qElemType2 = !quant.uniform<i4:f16, 1.1534313725490195>

// CHECK-LABEL: @ExpandZMajorConvChannelsI4Quant
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x30x30x!qElemType, {order = #NHWC}>)
func.func @ExpandZMajorConvChannelsI4Quant(%arg0: tensor<1x3x30x30x!qElemType1, {order = #NHWC}>) -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}> {
    %0 = const.Declare tensor<5x3x3x3x!qElemType, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16>, [
            #const.Reorder<#NHWC>,
            #const.CastElemType<si4>,
            #const.CastElemType<!qElemType>
    ]

    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType1, {order = #NHWC}>, tensor<5x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

    return %1 : tensor<1x5x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<32x32x3x3x!qElemType2, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x32x30x30x!qElemType, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 29, 0, 0]
    // CHECK-SAME:      output_padding = [0, 27, 0, 0]
    // CHECK-SAME:      -> tensor<1x32x28x28x!qElemType1, {order = #NHWC}>
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
!qElemType = !quant.uniform<u8:f16, 0.0078657811763239837:128>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.0078657811763239837:128>

// CHECK-LABEL: @ExpandAvgPool
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x3x513x513xf16, {order = #NHWC}>)
func.func @ExpandAvgPool(%arg0: tensor<1x3x513x513xf16, {order = #NHWC}>)
    -> tensor<1x3x513x513x!qElemType, {order = #NHWC}> {
    %AVGPOOL = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x3x513x513x!qElemType, {order = #NHWC}>

    return %AVGPOOL : tensor<1x3x513x513x!qElemType, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x513x513xf16, {order = #NHWC}>
    // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[EXPAND]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x513x513x!qElemType, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[AVGPOOL]] [0, 0, 0, 0] [1, 3, 513, 513]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandMaxPool
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x1x4xf16, {order = #NHWC}>)
func.func @ExpandMaxPool(%arg0: tensor<1x3x1x4xf16, {order = #NHWC}>) -> tensor<1x3x1x4xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>
        } : tensor<1x3x1x4xf16, {order = #NHWC}> -> tensor<1x3x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x3x1x4xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[EXPAND]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[MAXPOOL]] [0, 0, 0, 0] [1, 3, 1, 4]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwise
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x3x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x3x28x28xf16, {order = #NHWC}>)
func.func @ExpandEltwise(%arg0: tensor<1x3x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x3x28x28xf16, {order = #NHWC}>)
        -> tensor<1x3x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<> } :
        tensor<1x3x28x28xf16, {order = #NHWC}>, tensor<1x3x28x28xf16, {order = #NHWC}>
        -> tensor<1x3x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x3x28x28xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND_IN0:%.+]] = IE.Expand([[INPUT_0]])
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND_IN1:%.+]] = IE.Expand([[INPUT_1]])
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND_IN0]], [[EXPAND_IN1]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 3, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandGroupConv
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x40x80xf16, {order = #NHWC}>)
func.func @ExpandGroupConv(%arg0: tensor<1x3x40x80xf16, {order = #NHWC}>) -> tensor<1x3x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<3x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<3x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 3,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<3x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x3x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {order = #NHWC}>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[GROUP_CONV]] [0, 0, 0, 0] [1, 3, 37, 73]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AlignedOutGroupConvWithConsumer
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x40x80xf16, {order = #NHWC}>)
func.func @AlignedOutGroupConvWithConsumer(%arg0: tensor<1x3x40x80xf16, {order = #NHWC}>) -> tensor<1x3x34x66xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<3x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<3x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 3,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<3x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x3x37x73xf16, {order = #NHWC}>

    %1 = IE.GroupConvolution(%0, %weights) {
        dilations = [1, 1],
        groups = 3,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
    } : tensor<1x3x37x73xf16, {order = #NHWC}>, tensor<3x1x4x8xf16, {order = #NHWC}>
        -> tensor<1x3x34x66xf16, {order = #NHWC}>

    return %1 : tensor<1x3x34x66xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {order = #NHWC}>
    // CHECK:       [[GROUP_CONV_0:%.+]] = IE.GroupConvolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>
    // CHECK:       [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[GROUP_CONV_0]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x34x66xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[GROUP_CONV_1]] [0, 0, 0, 0] [1, 3, 34, 66]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoAlignedOutGroupConvWithConsumer
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x40x80xf16, {order = #NHWC}>)
func.func @NoAlignedOutGroupConvWithConsumer(%arg0: tensor<1x3x40x80xf16, {order = #NHWC}>) -> tensor<1x3x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<3x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<3x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 3,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<3x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x3x37x73xf16, {order = #NHWC}>

    %1 = IE.Swish(%0) : tensor<1x3x37x73xf16, {order = #NHWC}> -> tensor<1x3x37x73xf16, {order = #NHWC}>

    return %1 : tensor<1x3x37x73xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {order = #NHWC}>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 13, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[GROUP_CONV]] [0, 0, 0, 0] [1, 3, 37, 73]
    // CHECK:       [[SWISH:%.+]] = IE.Swish([[SLICE]])
    // CHECK-NEXT:  return [[SWISH]]
}
