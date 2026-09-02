//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% num-of-dpu-groups=1 allow-custom-values=true enable-auto-padding-odu enable-auto-padding-idu" --expand-activation-channels --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

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
// CHECK:       [[SW_AND:%.+]] = IE.And([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-NOT:   IE.Slice
// CHECK:       return [[SW_AND]] : tensor<1x3x30x25xf16, {order = #NHWC}>

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
func.func @ExpandInterpolateChannels(%input: tensor<1x21x48x48xf16, {order = #NHWC}>) -> tensor<1x21x336x336xf16, {order = #NHWC}> {
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
                                   scales_attr = [7.000000e+00, 7.000000e+00],
                                   sizes_attr = [336, 336]} :
        tensor<1x21x48x48xf16, {order = #NHWC}>
        -> tensor<1x21x336x336xf16, {order = #NHWC}>
    return %interp : tensor<1x21x336x336xf16, {order = #NHWC}>

    // CHECK:    [[EXPAND:%.+]] = IE.Expand([[INPUT]])

    // CHECK:    [[ADD:%.+]] = IE.Add([[EXPAND]], [[EXPAND]])
    // CHECK-SAME:        -> tensor<1x32x48x48xf16, {order = #NHWC}>

    // CHECK:    [[INTERP:%.+]] = IE.Interpolate([[ADD]])
    // CHECK-SAME:        -> tensor<1x32x336x336xf16, {order = #NHWC}>

    // CHECK:    [[SLICE:%.+]] = IE.Slice [[INTERP]]
    // CHECK-SAME:        to tensor<1x21x336x336xf16, {order = #NHWC}>

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

// -----

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:128>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @ExpandChannelsWithAvgPoolOp
func.func @ExpandChannelsWithAvgPoolOp(%arg0: tensor<1x1079x?x12xf16, {bounds = #const.OpaqueI64Elements<[1, 1079, 319, 12]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 2158, 638]> : tensor<4xsi64>, order = #NHWC}> {
  %0 = IE.PermuteCast(%arg0) {
    dst_order = #NHWC,
    mem_perm = #NCHW
  } : tensor<1x1079x?x12xf16, {bounds = #const.OpaqueI64Elements<[1, 1079, 319, 12]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  %1 = IE.AvgPool(%0) {
    exclude_pads,
    kernel_size = [1, 1],
    pads_begin = [0, 0],
    pads_end = [0, 0],
    rounding_type = #IE.rounding_type<FLOOR>,
    strides = [1, 1]
  } : tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  %2 = IE.DepthToSpace(%1) {
    block_size = 2 : i64,
    mode = #IE.depth_to_space_mode<DEPTH_FIRST>
  } : tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x3x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
  return %2 : tensor<1x3x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       [[PERM:%.+]] = IE.PermuteCast([[INPUT:%arg[0-9]]])
  // CHECK-SAME:      -> tensor<1x12x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       [[EXPAND:%.+]] = IE.Expand([[PERM]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 4, 0, 0]}
  // CHECK-SAME:      -> tensor<1x16x1079x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[EXPAND]])
  // CHECK-SAME:      input_padding = [0, 4, 0, 0]
  // CHECK-SAME:      output_padding = [0, 4, 0, 0]
  // CHECK-SAME:      -> tensor<1x16x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 16, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       [[SLICE:%.+]] = IE.Slice [[AVGPOOL]] [0, 0, 0, 0] [1, 12, 1079, -9223372036854775808]
  // CHECK-SAME:      to tensor<1x12x1079x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 12, 1079, 319]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       [[D2S:%.+]] = IE.DepthToSpace([[SLICE]])
  // CHECK-SAME:      -> tensor<1x3x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
  // CHECK:       return [[D2S]] : tensor<1x3x2158x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 3, 2158, 638]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandEltwiseAddChannels
// CHECK-SAME:  [[ARG_0:%[^:]+]]: tensor<1x5x16x16xf16, {order = #NHWC}>
// CHECK-SAME:  [[ARG_1:%[^:]+]]: tensor<1x5x16x16xf16, {order = #NHWC}>
func.func @ExpandEltwiseAddChannels(%arg0: tensor<1x5x16x16xf16, {order = #NHWC}>,
                                    %arg1: tensor<1x5x16x16xf16, {order = #NHWC}>)
        -> tensor<1x5x16x16xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
        tensor<1x5x16x16xf16, {order = #NHWC}>, tensor<1x5x16x16xf16, {order = #NHWC}>
        -> tensor<1x5x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x5x16x16xf16, {order = #NHWC}>

  // IE.Add is performed at the aligned shape, then sliced back to 5 channels.
  // CHECK:   [[EXP_0:%.+]] = IE.Expand([[ARG_0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, {{[0-9]+}}, 0, 0]}
  // CHECK:   [[EXP_1:%.+]] = IE.Expand([[ARG_1]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, {{[0-9]+}}, 0, 0]}
  // CHECK:   [[ADD:%.+]] = IE.Add([[EXP_0]], [[EXP_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
  // CHECK-SAME:   input_padding = [0, {{[0-9]+}}, 0, 0], output_padding = [0, {{[0-9]+}}, 0, 0]
  // CHECK:   [[SLICE:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 5, 16, 16]
  // CHECK:   return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!QuantileType.quantile<ui4:si8, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16:0, {4.343750e+00,3.625000e+00,-5.031250e+00,-1.375000e+00,-3.484375,4.156250e+00,-2.687500e+00,3.656250e+00,2.765625,-3.062500e+00,-1.640625,-6.281250e+00,-9.625000e+00,-1.7265625,7.937500e+00,5.906250e+00}>

// CHECK: [[QTYPE:!.+]] = !quant.uniform<!QuantileType.quantile<ui4:si8,
// CHECK: @ExpandConvolutionChannelsWithQuantileTypeFilter
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x4x1x1xf16, {order = #NHWC}>
// CHECK-SAME:  [[WEIGHTS:%.+]]: tensor<16x4x1x1x[[QTYPE]]>)
func.func @ExpandConvolutionChannelsWithQuantileTypeFilter(
        %arg0: tensor<1x4x1x1xf16, {order = #NHWC}>,
        %arg1: tensor<16x4x1x1x!qElemType>
) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
    %0 = IE.PermuteCast(%arg1) {dst_order = #NHWC, mem_perm = #NHWC}
        : tensor<16x4x1x1x!qElemType>
        -> tensor<16x4x1x1x!qElemType, {order = #NHWC}>
    %1 = IE.Convolution(%arg0, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x4x1x1xf16, {order = #NHWC}>,
        tensor<16x4x1x1x!qElemType, {order = #NHWC}>
        -> tensor<1x16x1x1xf16, {order = #NHWC}>
    return %1 : tensor<1x16x1x1xf16, {order = #NHWC}>

    // Activation is expanded from 4 to 16 channels.

    // A zero constant pads the filter from 4 to 16 input channels.
    // CHECK:       [[ZERO_CST:%.+]] = const.Declare tensor<16x12x1x1x[[QTYPE]], {order = #NHWC}> = dense<0> : tensor<16x12x1x1xui8, {order = #NHWC}>
    // CHECK:       [[FILTER_NHWC:%.+]] = IE.PermuteCast([[WEIGHTS]])
    // CHECK-SAME:      -> tensor<16x4x1x1x[[QTYPE]], {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:      -> tensor<1x16x1x1xf16, {order = #NHWC}>

    // CHECK:       [[EXPANDED_FILTER:%.+]] = IE.Concat([[FILTER_NHWC]], [[ZERO_CST]])
    // CHECK-SAME{{LITERAL}}:      static_offsets = [[0, 0, 0, 0], [0, 4, 0, 0]]
    // CHECK-SAME:      -> tensor<16x16x1x1x[[QTYPE]], {order = #NHWC}>

    // Convolution uses auto-padding attributes instead of inserting Slice on output.
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[EXPAND]], [[EXPANDED_FILTER]])
    // CHECK-SAME:      input_padding = [0, 12, 0, 0]
    // CHECK-SAME:      output_padding = [0, 0, 0, 0]
    // CHECK-SAME:      -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[CONV]]
}
