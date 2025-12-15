//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% num-of-dpu-groups=1 allow-custom-values=true enable-auto-padding-odu enable-auto-padding-idu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandChannelsWithReduceOp
module @ExpandChannelsWithReduceOp {

    config.PipelineOptions @Options {
        config.Option @config.ReduceSupported : true
    }

    // CHECK-LABEL:    func.func @ExpandConvolutionChannelsWithReduceMean
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x30x30xf16, {order = #NHWC}>)
    func.func @ExpandConvolutionChannelsWithReduceMean(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
        %filter = const.Declare tensor<5x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x1x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16, {order = #NHWC}>

        %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<5x1x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
        return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
        // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
        // CHECK:       [[MEAN:%.+]] = IE.ReduceMean([[EXPAND]])
        // CHECK-SAME:      input_padding = [0, 12, 0, 0]
        // CHECK-SAME:      output_padding = [0, 15, 0, 0]
        // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[MEAN]], [[CST]])
        // CHECK-SAME:      input_padding = [0, 15, 0, 0]
        // CHECK-SAME:      output_padding = [0, 11, 0, 0]
        // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
        // CHECK-NEXT:  return [[SLICE]]
    }

    // CHECK-LABEL:    func.func @ExpandConvolutionChannelsWithReduceSum
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x30x30xf16, {order = #NHWC}>)
    func.func @ExpandConvolutionChannelsWithReduceSum(%arg0: tensor<1x4x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
        %filter = const.Declare tensor<5x1x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x1x3x3xf16>, [#const.Reorder<#NHWC>]
        %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x4x30x30xf16, {order = #NHWC}> -> tensor<1x1x30x30xf16, {order = #NHWC}>

        %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x1x30x30xf16, {order = #NHWC}>, tensor<5x1x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
        return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
        // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
        // CHECK:       [[MEAN:%.+]] = IE.ReduceSum([[EXPAND]])
        // CHECK-SAME:      input_padding = [0, 12, 0, 0]
        // CHECK-SAME:      output_padding = [0, 15, 0, 0]
        // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
        // CHECK:       [[CONV:%.+]] = IE.Convolution([[MEAN]], [[CST]])
        // CHECK-SAME:      input_padding = [0, 15, 0, 0]
        // CHECK-SAME:      output_padding = [0, 11, 0, 0]
        // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
        // CHECK-NEXT:  return [[SLICE]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandConvolutionChannelsWithMultiply
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @ExpandConvolutionChannelsWithMultiply(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Multiply(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x3x30x30xf16, {order = #NHWC}>
        -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[INPUT]], [[INPUT]])
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[MULTIPLY]])
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

// CHECK-LABEL: @NoExpandConvolutionChannelsWithSubtract
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x3x30x30xf16, {order = #NHWC}>)
func.func @NoExpandConvolutionChannelsWithSubtract(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Subtract(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x3x30x30xf16, {order = #NHWC}>
        -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[EXPAND]], [[EXPAND]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[SUBTRACT]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChannelExpandAndOp
// CHECK-SAME:     ([[ARG0:%.+]]: tensor<1x3x30x25xf16, {order = #NHWC}>)
func.func @NoChannelExpandAndOp(%arg0: tensor<1x3x30x25xf16, {order = #NHWC}>) -> tensor<1x3x30x25xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x3x30x25xf16, {order = #NHWC}>, tensor<1x3x30x25xf16, {order = #NHWC}> -> tensor<1x3x30x25xf16, {order = #NHWC}>
    return %0 : tensor<1x3x30x25xf16, {order = #NHWC}>
}

// CHECK-NOT:   IE.Expand
// CHECK:       [[SW_AND:%.*]] = IE.And([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-NOT:   IE.Slice
// CHECK:       return [[SW_AND]] : tensor<1x3x30x25xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
// CHECK:  !qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
// CHECK:  !qElemType1 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127}>
// CHECK:  !qElemType2 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127,9.8713551919291337E-4:127}>
// CHECK:  !qElemType3 = !quant.uniform<u8<0:254>:f16:0, {8.7179349163385824E-4:127,5.2096149114173233E-4:127,0.0013264333169291339:127,5.0750492125984249E-4:127,9.8713551919291337E-4:127}>
// CHECK:  !qElemType4 = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127,6.8580447219488186E-4:127}>

// CHECK-LABEL: func.func @ExpandQuantConvolutionChannels
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x30x30x!qElemType, {order = #NHWC}>
func.func @ExpandQuantConvolutionChannels(%input: tensor<1x3x30x30x!qElemType, {order = #NHWC}>)
            -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3x!qElemType2, {order = #NHWC}> =
        dense<1.0> : tensor<5x3x3x3xf16, {order = #NHWC}>, [
        #const.CastElemType<ui8>,
        #const.CastElemType<!qElemType2>
    ]
    %1 = IE.Convolution(%input, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30x!qElemType, {order = #NHWC}>, tensor<5x3x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x5x28x28x!qElemType1, {order = #NHWC}>
    return %1 : tensor<1x5x28x28x!qElemType1, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3x!qElemType2, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30x!qElemType, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28x!qElemType4, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ExpandConvolutionChannelsWithAdd
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x30x30xf16, {order = #NHWC}>
func.func @ExpandConvolutionChannelsWithAdd(%arg0: tensor<1x3x30x30xf16, {order = #NHWC}>) -> tensor<1x5x28x28xf16, {order = #NHWC}> {
    %filter = const.Declare tensor<5x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<5x3x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<1x3x30x30xf16, {order = #NHWC}>
        -> tensor<1x3x30x30xf16, {order = #NHWC}>
    %1 = IE.Convolution(%0, %filter) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x30x30xf16, {order = #NHWC}>, tensor<5x3x3x3xf16, {order = #NHWC}> -> tensor<1x5x28x28xf16, {order = #NHWC}>
    return %1 : tensor<1x5x28x28xf16, {order = #NHWC}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]])
    // CHECK-SAME:      -> tensor<1x16x30x30xf16, {order = #NHWC}>
    // CHECK:       [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ADD]], [[CST]])
    // CHECK-SAME:      input_padding = [0, 13, 0, 0]
    // CHECK-SAME:      output_padding = [0, 11, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x28x28xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 5, 28, 28]
    // CHECK-NEXT:  return [[SLICE]]
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

// CHECK-LABEL: @ExpandInterpolateChannels
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x21x48x48xf16, {order = #NHWC}>)
func.func @ExpandInterpolateChannels(%input: tensor<1x21x48x48xf16, {order = #NHWC}>) -> tensor<1x21x384x384xf16, {order = #NHWC}> {
    %add = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
        tensor<1x21x48x48xf16, {order = #NHWC}>,
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x48x48xf16, {order = #NHWC}>
    %interp = IE.Interpolate(%add) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                                   shape_calc_mode = <SIZES>,
                                   coord_mode = <ALIGN_CORNERS>,
                                   nearest_mode = <ROUND_PREFER_CEIL>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [8.000000e+00, 8.000000e+00],
                                   sizes_attr = [384, 384]} :
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x384x384xf16, {order = #NHWC}>
    return %interp : tensor<1x21x384x384xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]])

    // CHECK:    [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x32x48x48xf16, {order = #NHWC}>

    // CHECK:    [[INTERP:%.+]] = IE.Interpolate([[ADD]])
    // CHECK-SAME:        -> tensor<1x32x384x384xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[INTERP]]
    // CHECK-SAME:        to tensor<1x21x384x384xf16, {order = #NHWC}>

    // CHECK:    return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoExpandInterpolateChannelsNoDPUOpBefore
func.func @NoExpandInterpolateChannelsNoDPUOpBefore(%input: tensor<1x21x48x48xf16, {order = #NHWC}>) -> tensor<1x21x384x384xf16, {order = #NHWC}> {
    %interp = IE.Interpolate(%input) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                                   shape_calc_mode = <SIZES>,
                                   coord_mode = <ALIGN_CORNERS>,
                                   nearest_mode = <ROUND_PREFER_CEIL>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [8.000000e+00, 8.000000e+00],
                                   sizes_attr = [384, 384]} :
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x384x384xf16, {order = #NHWC}>
    return %interp : tensor<1x21x384x384xf16, {order = #NHWC}>

    // CHECK-NOT: IE.Expand
    // CHECK-NOT: IE.Slice
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoExpandInterpolateChannelsBigSize
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x21x64x64xf16, {order = #NHWC}>)
func.func @NoExpandInterpolateChannelsBigSize(%input: tensor<1x21x64x64xf16, {order = #NHWC}>) -> tensor<1x21x512x512xf16, {order = #NHWC}> {
    %add = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
        tensor<1x21x64x64xf16, {order = #NHWC}>,
        tensor<1x21x64x64xf16, {order = #NHWC}>
        -> tensor<1x21x64x64xf16, {order = #NHWC}>
    %interp = IE.Interpolate(%add) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                                   shape_calc_mode = <SIZES>,
                                   coord_mode = <ALIGN_CORNERS>,
                                   nearest_mode = <ROUND_PREFER_CEIL>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [8.000000e+00, 8.000000e+00],
                                   sizes_attr = [512, 512]} :
        tensor<1x21x64x64xf16, {order = #NHWC}>
        -> tensor<1x21x512x512xf16, {order = #NHWC}>
    return %interp : tensor<1x21x512x512xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]])

    // CHECK:    [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x32x64x64xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[ADD]]
    // CHECK-SAME:        to tensor<1x21x64x64xf16, {order = #NHWC}>

    // CHECK:    [[INTERP:%.+]] = IE.Interpolate([[SLICE]])
    // CHECK-SAME:        -> tensor<1x21x512x512xf16, {order = #NHWC}>

    // CHECK:    return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoExpandInterpolateChannelsWrongAxes
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x21x48x48xf16, {order = #NHWC}>)
func.func @NoExpandInterpolateChannelsWrongAxes(%input: tensor<1x21x48x48xf16, {order = #NHWC}>) -> tensor<1x21x384x48xf16, {order = #NHWC}> {
    %add = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
        tensor<1x21x48x48xf16, {order = #NHWC}>,
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x48x48xf16, {order = #NHWC}>
    %interp = IE.Interpolate(%add) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                                   shape_calc_mode = <SIZES>,
                                   coord_mode = <ALIGN_CORNERS>,
                                   nearest_mode = <ROUND_PREFER_CEIL>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [8.000000e+00],
                                   sizes_attr = [384]} :
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x384x48xf16, {order = #NHWC}>
    return %interp : tensor<1x21x384x48xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]])

    // CHECK:    [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x32x48x48xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[ADD]]
    // CHECK-SAME:        to tensor<1x21x48x48xf16, {order = #NHWC}>

    // CHECK:    [[INTERP:%.+]] = IE.Interpolate([[SLICE]])
    // CHECK-SAME:        -> tensor<1x21x384x48xf16, {order = #NHWC}>

    // CHECK:    return [[INTERP]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoExpandInterpolateChannelsTooSmall
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x48x48xf16, {order = #NHWC}>)
func.func @NoExpandInterpolateChannelsTooSmall(%input: tensor<1x2x48x48xf16, {order = #NHWC}>) -> tensor<1x2x384x384xf16, {order = #NHWC}> {
    %add = IE.Add(%input, %input) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
        tensor<1x2x48x48xf16, {order = #NHWC}>,
        tensor<1x2x48x48xf16, {order = #NHWC}>
        -> tensor<1x2x48x48xf16, {order = #NHWC}>
    %interp = IE.Interpolate(%add) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>,
                                   shape_calc_mode = <SIZES>,
                                   coord_mode = <ALIGN_CORNERS>,
                                   nearest_mode = <ROUND_PREFER_CEIL>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
                                   axes_attr = [2, 3],
                                   operandSegmentSizes = array<i32: 1, 0, 0, 0>,
                                   scales_attr = [8.000000e+00, 8.000000e+00],
                                   sizes_attr = [384, 384]} :
        tensor<1x2x48x48xf16, {order = #NHWC}>
        -> tensor<1x2x384x384xf16, {order = #NHWC}>
    return %interp : tensor<1x2x384x384xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]])

    // CHECK:    [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x16x48x48xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[ADD]]
    // CHECK-SAME:        to tensor<1x2x48x48xf16, {order = #NHWC}>

    // CHECK:    [[INTERP:%.+]] = IE.Interpolate([[SLICE]])
    // CHECK-SAME:        -> tensor<1x2x384x384xf16, {order = #NHWC}>

    // CHECK:    return [[INTERP]]
}
