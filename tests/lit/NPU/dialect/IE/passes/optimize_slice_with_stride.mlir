//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-slice-with-stride %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSliceToConvFromConvert
func.func @ConvertSliceToConvFromConvert(%arg0: tensor<1x3x1088x1920xf32, {order = #NHWC}>)
    -> tensor<1x1x1088x1920xf16, {order = #NHWC}> {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1088x1920xf32, {order = #NHWC}> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONVERT [0, 1, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16, {order = #NHWC}> to tensor<1x1x1088x1920xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[CONVERT_INPUT:%.*]] = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x1088x1920xf32, {order = #NHWC}> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.ShapeCast {shape = [1, 48, 1088, 120]} inputs([[CONVERT_INPUT]] : tensor<1x3x1088x1920xf16, {order = #NHWC}>) -> tensor<1x48x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<16x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C{{([0]{181})}}
    // CHECK-SAME:      0000003C0000"> : tensor<16x48x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x1088x120xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<16x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x16x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.ShapeCast {shape = [1, 1, 1088, 1920]} inputs([[CONV]] : tensor<1x16x1088x120xf16, {order = #NHWC}>) -> tensor<1x1x1088x1920xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x1x1088x1920xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSliceWithSmallChannel
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x2x272x480xf32, {order = #NHWC}>
func.func @ConvertSliceWithSmallChannel(%arg0: tensor<1x2x272x480xf32, {order = #NHWC}>)
    -> tensor<1x1x272x480xf16, {order = #NHWC}> {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x272x480xf32, {order = #NHWC}> -> tensor<1x2x272x480xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONVERT [0, 1, 0, 0] [1, 1, 272, 480] : tensor<1x2x272x480xf16, {order = #NHWC}> to tensor<1x1x272x480xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x272x480xf16, {order = #NHWC}>


    // CHECK:   [[CONVERT_INPUT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16} : tensor<1x2x272x480xf32, {order = #NHWC}> -> tensor<1x2x272x480xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.ShapeCast {shape = [1, 32, 272, 30]} inputs([[CONVERT_INPUT]] : tensor<1x2x272x480xf16, {order = #NHWC}>) -> tensor<1x32x272x30xf16, {order = #NHWC}>
    // CHECK:   [[WEIGHTS:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C{{([0]{128})}}
    // CHECK-SAME:      0000003C"> : tensor<16x32x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x272x30xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<16x32x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x272x30xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.ShapeCast {shape = [1, 1, 272, 480]} inputs([[CONV]] : tensor<1x16x272x30xf16, {order = #NHWC}>) -> tensor<1x1x272x480xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x1x272x480xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertSliceIfChannelNotMeetRequirement
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x2x80x80xf32, {order = #NHWC}>
func.func @NotConvertSliceIfChannelNotMeetRequirement(%arg0: tensor<1x2x80x80xf32, {order = #NHWC}>)
    -> tensor<1x1x80x80xf16, {order = #NHWC}> {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x80x80xf32, {order = #NHWC}> -> tensor<1x2x80x80xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONVERT [0, 1, 0, 0] [1, 1, 80, 80] : tensor<1x2x80x80xf16, {order = #NHWC}> to tensor<1x1x80x80xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x80x80xf16, {order = #NHWC}>

    // CHECK:   [[CONVERT_INPUT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16} : tensor<1x2x80x80xf32, {order = #NHWC}> -> tensor<1x2x80x80xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.+]]  = IE.Slice [[CONVERT_INPUT]] [0, 1, 0, 0] [1, 1, 80, 80] : tensor<1x2x80x80xf16, {order = #NHWC}> to tensor<1x1x80x80xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x1x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertSliceToConvFromPermuteCast
func.func @ConvertSliceToConvFromPermuteCast(%arg0: tensor<1x1088x1920x3xf16>)
    -> tensor<1x1x1088x1920xf16, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3xf16> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16, {order = #NHWC}> to tensor<1x1x1088x1920xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST_INPUT:%.*]]  = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3xf16> -> tensor<1x3x1088x1920xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.ShapeCast {shape = [1, 48, 1088, 120]} inputs([[PERMUTECAST_INPUT]] : tensor<1x3x1088x1920xf16, {order = #NHWC}>) -> tensor<1x48x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[WEIGHTS:%.*]] = const.Declare tensor<16x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK-SAME:      0000003C{{([0]{196})}}
    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x1088x120xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<16x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x16x1088x120xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.ShapeCast {shape = [1, 1, 1088, 1920]} inputs(%2 : tensor<1x16x1088x120xf16, {order = #NHWC}>) -> tensor<1x1x1088x1920xf16, {order = #NHWC}>
    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x1x1088x1920xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipSliceNCHW
func.func @SkipSliceNCHW(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x1x1088x1920xf16> {
    %SLICE = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x1x1088x1920xf16>
    return %SLICE : tensor<1x1x1088x1920xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x1x1088x1920xf16>
    // CHECK:   return [[SLICE]] : tensor<1x1x1088x1920xf16>

}

// -----

// CHECK-LABEL: @SkipSliceOnHeight
func.func @SkipSliceOnHeight(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x3x100x1920xf16> {
    %SLICE = IE.Slice %arg0 [0, 0, 3, 0] [1, 3, 100, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x3x100x1920xf16>
    return %SLICE : tensor<1x3x100x1920xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 3, 0] [1, 3, 100, 1920] : tensor<1x3x1088x1920xf16> to tensor<1x3x100x1920xf16>
    // CHECK:   return [[SLICE]] : tensor<1x3x100x1920xf16>

}

// -----

// CHECK-LABEL: @SkipSliceOnWidth
func.func @SkipSliceOnWidth(%arg0: tensor<1x3x1088x1920xf16>) -> tensor<1x3x1088x100xf16> {
    %SLICE = IE.Slice %arg0 [0, 0, 0, 3] [1, 3, 1088, 100] : tensor<1x3x1088x1920xf16> to tensor<1x3x1088x100xf16>
    return %SLICE : tensor<1x3x1088x100xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice %arg0 [0, 0, 0, 3] [1, 3, 1088, 100] : tensor<1x3x1088x1920xf16> to tensor<1x3x1088x100xf16>
    // CHECK:   return [[SLICE]] : tensor<1x3x1088x100xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-LABEL: @SkipConvertSliceToConvFromPermuteCastWithQuantizedType
func.func @SkipConvertSliceToConvFromPermuteCastWithQuantizedType(%arg0: tensor<1x1088x1920x3x!qElemType>)
    -> tensor<1x1x1088x1920x!qElemType, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3x!qElemType> -> tensor<1x3x1088x1920x!qElemType, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920x!qElemType, {order = #NHWC}> to tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>

    return %SLICE : tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x1088x1920x3x!qElemType> -> tensor<1x3x1088x1920x!qElemType, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[PERMUTECAST]] [0, 3, 0, 0] [1, 1, 1088, 1920] : tensor<1x3x1088x1920x!qElemType, {order = #NHWC}> to tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>
    // CHECK:   return [[SLICE]] : tensor<1x1x1088x1920x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NotConvertIfFitCmx
func.func @NotConvertIfFitCmx(%arg0: tensor<1x16x16x3xf16>)
    -> tensor<1x1x16x16xf16, {order = #NHWC}> {
    %PERMUTECAST = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x3xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    %SLICE = IE.Slice %PERMUTECAST [0, 3, 0, 0] [1, 1, 16, 16] : tensor<1x3x16x16xf16, {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x16x16xf16, {order = #NHWC}>

    // CHECK:   [[PERMUTECAST:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x16x16x3xf16> -> tensor<1x3x16x16xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice [[PERMUTECAST]] [0, 3, 0, 0] [1, 1, 16, 16] : tensor<1x3x16x16xf16, {order = #NHWC}> to tensor<1x1x16x16xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x1x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @OptimizeSliceConcat
func.func @OptimizeSliceConcat(%arg0: tensor<1x1024x32x32xf16, {order = #NHWC}>) -> tensor<1x512x32x32xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x1x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %WEIGHTS2 = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<512x1024x3x3xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 0, 0, 0] [1, 511, 32, 32] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x511x32x32xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 511, 0, 0]]} : tensor<1x511x32x32xf16, {order = #NHWC}>, tensor<1x1x32x32xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %CONV_OUT = IE.Convolution(%CONCAT, %WEIGHTS2) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]} : tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<512x512x3x3xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>

    return %CONV_OUT : tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [511, 1024, 3, 3]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [1, 0, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[CONV_IN:%.+]] = IE.Convolution(%arg0, [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<512x1024x3x3xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<1x512x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 511, 0, 0], [0, 0, 0, 0]>]
    // CHECK:   [[ADD:%.+]] = IE.Add([[CONV_IN]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
    // CHECK-SAME:      tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x512x32x32xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK:   [[CONV_OUT:%.+]] = IE.Convolution([[ADD]], [[WEIGHTS2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<512x512x3x3xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK:   return [[CONV_OUT]] : tensor<1x512x32x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceConcatWithTwoUsers
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x32x32xf16, {order = #NHWC}
func.func @NotOptimizeSliceConcatWithTwoUsers(%arg0: tensor<1x1024x32x32xf16, {order = #NHWC}>)
                            -> (tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x511x16x64xf16, {order = #NHWC}>) {
    %WEIGHTS = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x1x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %WEIGHTS2 = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS) {
                dilations = [1, 1],
                pads_begin = [1, 1], pads_end = [1, 1],
                strides = [1, 1]
            } : tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<512x1024x3x3xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 0, 0, 0] [1, 511, 32, 32] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x511x32x32xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {
                static_offsets = [[0, 0, 0, 0], [0, 511, 0, 0]]} : tensor<1x511x32x32xf16, {order = #NHWC}>, tensor<1x1x32x32xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %RESHAPE = IE.AffineReshape(%SLICE) {
                dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 511, 16, 64] } : tensor<1x511x32x32xf16, {order = #NHWC}> -> tensor<1x511x16x64xf16, {order = #NHWC}>

    return %CONCAT, %RESHAPE : tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x511x16x64xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:                  dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
    // CHECK:   [[SLICE:%.+]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 511, 32, 32] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x511x32x32xf16, {order = #NHWC}>

    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {
    // CHECK-SAME{LITERAL}:                  static_offsets = [[0, 0, 0, 0], [0, 511, 0, 0]]} : tensor<1x511x32x32xf16, {order = #NHWC}>, tensor<1x1x32x32xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME{LITERAL}:                  dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 511, 16, 64]} : tensor<1x511x32x32xf16, {order = #NHWC}> -> tensor<1x511x16x64xf16, {order = #NHWC}>

    // CHECK:   return [[CONCAT]], [[RESHAPE]] : tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x511x16x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceConcatIfNotLowestDim
func.func @NotOptimizeSliceConcatIfNotLowestDim(%arg0: tensor<1x1024x32x32xf16, {order = #NHWC}>) -> tensor<1x512x32x32xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x512x32x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %WEIGHTS2 = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<512x1024x3x3xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 0, 0, 0] [1, 512, 32, 30] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x512x32x30xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 30]]} : tensor<1x512x32x30xf16, {order = #NHWC}>, tensor<1x512x32x2xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %CONV_OUT = IE.Convolution(%CONCAT, %WEIGHTS2) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]} : tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<512x512x3x3xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>

    return %CONV_OUT : tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<512x1024x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x1024x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x512x32x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = const.Declare tensor<512x512x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<512x512x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[CONV_IN:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<512x1024x3x3xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK-NOT:   IE.Add
    // CHECK:       [[SLICE:%.*]] = IE.Slice [[CONV_IN]] [0, 0, 0, 0] [1, 512, 32, 30] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x512x32x30xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.*]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 30]]} :
    // CHECK-SAME:      tensor<1x512x32x30xf16, {order = #NHWC}>, tensor<1x512x32x2xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK:       [[CONV_OUT:%.*]] = IE.Convolution([[CONCAT]], [[WEIGHTS2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<512x512x3x3xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK:   return [[CONV_OUT]] : tensor<1x512x32x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotOptimizeSliceConcatIfSliceInputNotConv
func.func @NotOptimizeSliceConcatIfSliceInputNotConv(%arg0: tensor<1x512x32x32xf16, {order = #NHWC}>) -> tensor<1x512x32x32xf16, {order = #NHWC}> {
    %ADD_WEIGHTS = const.Declare tensor<1x512x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x32xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x512x32x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x512x32x32xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 512, 32, 30] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x512x32x30xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 30]]} : tensor<1x512x32x30xf16, {order = #NHWC}>, tensor<1x512x32x2xf16, {order = #NHWC}>
        -> tensor<1x512x32x32xf16, {order = #NHWC}>

    return %CONCAT : tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x512x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x32xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x512x32x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x512x32x2xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK:   [[ADD_IN:%.*]] = IE.Add(%arg0, [[ADD_WEIGHTS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:      tensor<1x512x32x32xf16, {order = #NHWC}>, tensor<1x512x32x32xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>
    // CHECK:   [[SLICE:%.*]] = IE.Slice %0 [0, 0, 0, 0] [1, 512, 32, 30] : tensor<1x512x32x32xf16, {order = #NHWC}> to tensor<1x512x32x30xf16, {order = #NHWC}>
    // CHECK-NOT:   IE.Add
    // CHECK:   [[CONCAT:%.*]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 30]]} :
    // CHECK-SAME:      tensor<1x512x32x30xf16, {order = #NHWC}>, tensor<1x512x32x2xf16, {order = #NHWC}> -> tensor<1x512x32x32xf16, {order = #NHWC}>

    // CHECK:   return [[CONCAT]] : tensor<1x512x32x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceIntoPreviousConv
func.func @FuseSliceIntoPreviousConv(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x8x256x256xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %BIAS = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x256x256xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 0, 0, 0] [1, 8, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x8x256x256xf16, {order = #NHWC}>

    return %SLICE : tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<8x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [8, 32, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x256x256xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<8x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x8x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseConvSliceWithNonBroadcastBias
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x32x90x160xf16, {order = #NHWC}>
func.func @FuseConvSliceWithNonBroadcastBias(%arg0: tensor<1x32x90x160xf16, {order = #NHWC}>) -> tensor<1x72x90x160xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x80x1x1xf16> = dense<1.250000e-01> : tensor<1x72x1x1xf32>, [#const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    %cst_1 = const.Declare tensor<80x32x3x3xf16, {order = #NHWC}> = dense<0.870000e-01> : tensor<72x24x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [8, 8, 0, 0]>]

    %8 = IE.Convolution(%arg0, %cst_1, %cst_0) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = -0.028176676481962204 : f64>,
            strides = [1, 1]} : tensor<1x32x90x160xf16, {order = #NHWC}>, tensor<80x32x3x3xf16, {order = #NHWC}>, tensor<1x80x1x1xf16> -> tensor<1x80x90x160xf16, {order = #NHWC}>
    %9 = IE.Slice %8 [0, 0, 0, 0] [1, 72, 90, 160] : tensor<1x80x90x160xf16, {order = #NHWC}> to tensor<1x72x90x160xf16, {order = #NHWC}>

    return %9 : tensor<1x72x90x160xf16, {order = #NHWC}>

    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x72x1x1xf16> = dense<1.250000e-01> : tensor<1x72x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<72x32x3x3xf16, {order = #NHWC}> = dense<8.700000e-02> : tensor<72x24x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    // CHECK-DAG:   [[CONV:%.+]] = IE.Convolution([[INPUT0]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = -0.028176676481962204 : f64>,
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x32x90x160xf16, {order = #NHWC}>, tensor<72x32x3x3xf16, {order = #NHWC}>, tensor<1x72x1x1xf16>
    // CHECK-SAME:      -> tensor<1x72x90x160xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x72x90x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceIntoPreviousConvWithOffset
func.func @FuseSliceIntoPreviousConvWithOffset(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x8x256x256xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %BIAS = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x256x256xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 1, 0, 0] [1, 8, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x8x256x256xf16, {order = #NHWC}>

    return %SLICE : tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<8x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.SubView<[1, 0, 0, 0], [8, 32, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CONV:%.+]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x256x256xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<8x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x8x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FuseSliceIntoPreviousConvWithBias
func.func @FuseSliceIntoPreviousConvWithBias(%arg0: tensor<1x32x256x256xf16, {order = #NHWC}>) -> tensor<1x8x256x256xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %BIAS = const.Declare tensor<1x32x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<1x32x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x256x256xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 1, 0, 0] [1, 8, 256, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x8x256x256xf16, {order = #NHWC}>

    return %SLICE : tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<8x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.SubView<[1, 0, 0, 0], [8, 32, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x8x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x32x1x1xf16>, [#const.SubView<[0, 1, 0, 0], [1, 8, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CONV:%.+]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x256x256xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<8x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x8x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x8x256x256xf16, {order = #NHWC}>

    // CHECK:   return [[CONV]] : tensor<1x8x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseSliceIntoPreviousConvDueTONotEfficient
func.func @NotFuseSliceIntoPreviousConvDueTONotEfficient(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x8x64x64xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %BIAS = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [1, 1], pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x64x64xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 1, 0, 0] [1, 8, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x8x64x64xf16, {order = #NHWC}>

    return %SLICE : tensor<1x8x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CONV:%.*]] = IE.Convolution(%arg0, [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [1, 1], pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x32x64x64xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<32x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x32x64x64xf16, {order = #NHWC}>
    // CHECK-DAG:   [[SLICE:%.*]] = IE.Slice [[CONV]] [0, 1, 0, 0] [1, 8, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x8x64x64xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x8x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSliceOC3ToConvsWithFactor16
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x640x640xf16, {order = #NHWC}>
func.func @ConvertSliceOC3ToConvsWithFactor16(%arg0: tensor<1x4x640x640xf16, {order = #NHWC}>)
    -> tensor<1x3x640x640xf16, {order = #NHWC}> {
    %SLICE = IE.Slice %arg0 [0, 0, 0, 0] [1, 3, 640, 640] : tensor<1x4x640x640xf16, {order = #NHWC}> to tensor<1x3x640x640xf16, {order = #NHWC}>

    return %SLICE : tensor<1x3x640x640xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 64, 640, 40]} inputs([[INPUT]] : tensor<1x4x640x640xf16, {order = #NHWC}>) -> tensor<1x64x640x40xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<48x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[IN_SHAPECAST]], [[WEIGHTS]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x64x640x40xf16, {order = #NHWC}>, tensor<48x64x1x1xf16, {order = #NHWC}> -> tensor<1x48x640x40xf16, {order = #NHWC}>
    // CHECK:       [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 3, 640, 640]} inputs([[CONV]] : tensor<1x48x640x40xf16, {order = #NHWC}>) -> tensor<1x3x640x640xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_SHAPECAST]] : tensor<1x3x640x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertSliceOC1ToConvsWithFactor16
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x640x640xf16, {order = #NHWC}
func.func @ConvertSliceOC1ToConvsWithFactor16(%arg0: tensor<1x4x640x640xf16, {order = #NHWC}>)
    -> tensor<1x1x640x640xf16, {order = #NHWC}> {
    %SLICE = IE.Slice %arg0 [0, 3, 0, 0] [1, 1, 640, 640] : tensor<1x4x640x640xf16, {order = #NHWC}> to tensor<1x1x640x640xf16, {order = #NHWC}>

    return %SLICE : tensor<1x1x640x640xf16, {order = #NHWC}>

    // CHECK:       [[IN_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 64, 640, 40]} inputs(%arg0 : tensor<1x4x640x640xf16, {order = #NHWC}>) -> tensor<1x64x640x40xf16, {order = #NHWC}>
    // CHECK:       [[WEIGHTS:%.*]] = const.Declare tensor<16x64x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[IN_SHAPECAST]], [[WEIGHTS]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x64x640x40xf16, {order = #NHWC}>, tensor<16x64x1x1xf16, {order = #NHWC}> -> tensor<1x16x640x40xf16, {order = #NHWC}>
    // CHECK:       [[OUT_SHAPECAST:%.*]] = IE.ShapeCast {shape = [1, 1, 640, 640]} inputs([[CONV]] : tensor<1x16x640x40xf16, {order = #NHWC}>) -> tensor<1x1x640x640xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_SHAPECAST]] : tensor<1x1x640x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipSliceLargeChannels
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x64x640x640xf16, {order = #NHWC}>
func.func @SkipSliceLargeChannels(%arg0: tensor<1x64x640x640xf16, {order = #NHWC}>)
    -> tensor<1x56x640x640xf16, {order = #NHWC}> {
    %SLICE = IE.Slice %arg0 [0, 0, 0, 0] [1, 56, 640, 640] : tensor<1x64x640x640xf16, {order = #NHWC}> to tensor<1x56x640x640xf16, {order = #NHWC}>

    return %SLICE : tensor<1x56x640x640xf16, {order = #NHWC}>

    // CHECK:       [[SLICE:%.*]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 56, 640, 640] : tensor<1x64x640x640xf16, {order = #NHWC}> to tensor<1x56x640x640xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x56x640x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseSliceWithConvOnMultiDims
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x128x387x387xf16, {order = #NHWC}>
func.func @NotFuseSliceWithConvOnMultiDims(%arg0: tensor<1x128x387x387xf16, {order = #NHWC}>)
    -> tensor<1x42x384x384xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<48x128x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<48x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %BIAS = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x48x1x1xf16>, [#const.Reorder<#NHWC>]
    %CONV = IE.Convolution(%arg0, %WEIGHTS, %BIAS) {
        dilations = [1, 1],
        pads_begin = [0, 0], pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x128x387x387xf16, {order = #NHWC}>, tensor<48x128x3x3xf16, {order = #NHWC}>, tensor<1x48x1x1xf16, {order = #NHWC}>
        -> tensor<1x48x385x385xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 0, 1, 1] [1, 42, 384, 384] : tensor<1x48x385x385xf16, {order = #NHWC}> to tensor<1x42x384x384xf16, {order = #NHWC}>

    return %SLICE : tensor<1x42x384x384xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<48x128x3x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<48x128x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x48x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CONV:%.*]] = IE.Convolution([[INPUT]], [[WEIGHTS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x128x387x387xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<48x128x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x48x385x385xf16, {order = #NHWC}>
    // CHECK-DAG:   [[SLICE:%.*]] = IE.Slice [[CONV]] [0, 0, 1, 1] [1, 42, 384, 384] : tensor<1x48x385x385xf16, {order = #NHWC}> to tensor<1x42x384x384xf16, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x42x384x384xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InsertIdentityConvOptimizeSliceConcat
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x32x32x32xf16, {order = #NHWC}>
func.func @InsertIdentityConvOptimizeSliceConcat(%arg0: tensor<1x32x32x32xf16, {order = #NHWC}>) -> tensor<1x48x32x32xf16, {order = #NHWC}> {
    %ADD_WEIGHTS = const.Declare tensor<1x32x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x32x32x32xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x24x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x1xf16>, [#const.Broadcast<3 : i64, 32 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 23, 0, 0]>]
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x32x32x32xf16, {order = #NHWC}>, tensor<1x32x32x32xf16, {order = #NHWC}> -> tensor<1x32x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 24, 32, 32] : tensor<1x32x32x32xf16, {order = #NHWC}> to tensor<1x24x32x32xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 24, 0, 0]]} : tensor<1x24x32x32xf16, {order = #NHWC}>, tensor<1x24x32x32xf16, {order = #NHWC}>
        -> tensor<1x48x32x32xf16, {order = #NHWC}>

    return %CONCAT : tensor<1x48x32x32xf16, {order = #NHWC}>



    // CHECK:   [[CST_1:%.+]] = const.Declare tensor<1x32x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x32x32x32xf16>, [#const.Reorder<#NHWC>]
    // CHECK:   [[ADD:%.+]] = IE.Add([[INPUT0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x32x32xf16, {order = #NHWC}>, tensor<1x32x32x32xf16, {order = #NHWC}> -> tensor<1x32x32x32xf16, {order = #NHWC}>
    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<48x32x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:              {{([003C00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{32})}}
    // CHECK-SAME:              {{([00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000]{15})}}
    // CHECK-SAME:              "> : tensor<48x32x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [24, 32, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [24, 0, 0, 0]>]
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[ADD]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x32x32xf16, {order = #NHWC}>, tensor<48x32x1x1xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>
    // CHECK:   [[CST:%.+]] = const.Declare tensor<1x48x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x1xf16>, [#const.Broadcast<3 : i64, 32 : i64>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 23, 0, 0]>, #const.PadWithZero<[0, 24, 0, 0], [0, 0, 0, 0]>]
    // CHECK:   [[ADD_OUT:%.+]] = IE.Add([[CONV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x48x32x32xf16, {order = #NHWC}>, tensor<1x48x32x32xf16, {order = #NHWC}> -> tensor<1x48x32x32xf16, {order = #NHWC}>

    // CHECK:   return [[ADD_OUT]] : tensor<1x48x32x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipInsertIdentityConvOptimizeSliceConcat
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1024x32x32xf16, {order = #NHWC}>
func.func @SkipInsertIdentityConvOptimizeSliceConcat(%arg0: tensor<1x1024x32x32xf16, {order = #NHWC}>) -> tensor<1x1024x32x32xf16, {order = #NHWC}> {
    %ADD_WEIGHTS = const.Declare tensor<1x1024x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1024x32x32xf16>, [#const.Reorder<#NHWC>]
    %CST_0 = const.Declare tensor<1x1x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<1x1024x32x32xf16, {order = #NHWC}> -> tensor<1x1024x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 1023, 32, 32] : tensor<1x1024x32x32xf16, {order = #NHWC}> to tensor<1x1023x32x32xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 1023, 0, 0]]} : tensor<1x1023x32x32xf16, {order = #NHWC}>, tensor<1x1x32x32xf16, {order = #NHWC}>
        -> tensor<1x1024x32x32xf16, {order = #NHWC}>

    return %CONCAT : tensor<1x1024x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1024x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1024x32x32xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x32x32xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<1x1x32x32xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[ADD:%.+]] = IE.Add([[INPUT0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x32x32xf16, {order = #NHWC}>, tensor<1x1024x32x32xf16, {order = #NHWC}> -> tensor<1x1024x32x32xf16, {order = #NHWC}>
    // CHECK-DAG:   [[SLICE:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 1023, 32, 32] : tensor<1x1024x32x32xf16, {order = #NHWC}> to tensor<1x1023x32x32xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1023, 0, 0]]} : tensor<1x1023x32x32xf16, {order = #NHWC}>, tensor<1x1x32x32xf16, {order = #NHWC}> -> tensor<1x1024x32x32xf16, {order = #NHWC}>

    // CHECK:   return [[CONCAT]] : tensor<1x1024x32x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.040966219060561235:139>

// CHECK-LABEL: @NoConvOptimizeSliceConcatQuant
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x32x64x64x!qElemType, {order = #NHWC}>
func.func @NoConvOptimizeSliceConcatQuant(%arg0: tensor<1x32x64x64x!qElemType, {order = #NHWC}>) -> tensor<1x32x64x64x!qElemType, {order = #NHWC}> {
    %CST_0 = const.Declare tensor<1x4x64x64x!qElemType, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x4x64x64xf32>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %SLICE = IE.Slice %arg0 [0, 0, 0, 0] [1, 28, 64, 64] : tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x28x64x64x!qElemType, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 28, 0, 0]]} : tensor<1x28x64x64x!qElemType, {order = #NHWC}>, tensor<1x4x64x64x!qElemType, {order = #NHWC}> -> tensor<1x32x64x64x!qElemType, {order = #NHWC}>

    return %CONCAT : tensor<1x32x64x64x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x4x64x64x!qElemType, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x4x64x64xf32>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[SLICE:%.+]] = IE.Slice [[INPUT0]] [0, 0, 0, 0] [1, 28, 64, 64] : tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x28x64x64x!qElemType, {order = #NHWC}>
    // CHECK-DAG:   [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 28, 0, 0]]} : tensor<1x28x64x64x!qElemType, {order = #NHWC}>, tensor<1x4x64x64x!qElemType, {order = #NHWC}> -> tensor<1x32x64x64x!qElemType, {order = #NHWC}>

    // CHECK:   return [[CONCAT]] : tensor<1x32x64x64x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoConvOptimizeSliceConcatForBlockArgument
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @NoConvOptimizeSliceConcatForBlockArgument(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x32x64x64xf16, {order = #NHWC}> {
    %CST_0 = const.Declare tensor<1x4x64x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x4x64x64xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %SLICE = IE.Slice %arg0 [0, 0, 0, 0] [1, 28, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x28x64x64xf16, {order = #NHWC}>
    %CONCAT = IE.Concat(%SLICE, %CST_0) {static_offsets = [[0, 0, 0, 0], [0, 28, 0, 0]]} : tensor<1x28x64x64xf16, {order = #NHWC}>, tensor<1x4x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    return %CONCAT : tensor<1x32x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x4x64x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x4x64x64xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[SLICE:%.+]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 28, 64, 64] : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x28x64x64xf16, {order = #NHWC}>
    // CHECK-DAG:   [[CONCAT:%.+]] = IE.Concat([[SLICE]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 28, 0, 0]]} : tensor<1x28x64x64xf16, {order = #NHWC}>, tensor<1x4x64x64xf16, {order = #NHWC}> -> tensor<1x32x64x64xf16, {order = #NHWC}>

    // CHECK:   return [[CONCAT]] : tensor<1x32x64x64xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.02578586316576191:128>
!qElemType1 = !quant.uniform<u8:f16:0, {8.0570125696705834E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128,8.7129681133756454E-4:128}>
!qElemType2 = !quant.uniform<i8:f16:1, {8.0570125696705834E-4,8.7129681133756454E-4}>
!qElemType3 = !quant.uniform<u8:f16:0, {8.0570125696705834E-4:128,8.7129681133756454E-4:128}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseSliceOneIntoPreviousPerAxisQuantizedConvWithOffset
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x160x1x1x!qElemType, {order = #NHWC}>
func.func @NotFuseSliceOneIntoPreviousPerAxisQuantizedConvWithOffset(%arg0: tensor<1x160x1x1x!qElemType, {order = #NHWC}>) -> tensor<1x1x1x1xf16, {order = #NHWC}> {
    %WEIGHTS = const.Declare tensor<16x160x1x1x!qElemType1, {order = #NHWC}> = dense<2.0> : tensor<1x2x1x160xf16>, [#const.CastElemType<!qElemType2>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [2, 160, 1, 1]>, #const.ConvertElemType<!qElemType3>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [14, 0, 0, 0]>]

    %CONV = IE.Convolution(%arg0, %WEIGHTS) {
        dilations = [1, 1],
        pads_begin = [0, 0], pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x160x1x1x!qElemType, {order = #NHWC}>, tensor<16x160x1x1x!qElemType1, {order = #NHWC}> -> tensor<1x16x1x1xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONV [0, 1, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
    return %SLICE : tensor<1x1x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x160x1x1x!qElemType1, {order = #NHWC}> = dense<2.000000e+00> :
    // CHECK-SAME{LITERAL}:    tensor<1x2x1x160xf16>, [#const.CastElemType<!qElemType2>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [2, 160, 1, 1]>, #const.ConvertElemType<!qElemType3>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [14, 0, 0, 0]>]

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } : tensor<1x160x1x1x!qElemType, {order = #NHWC}>, tensor<16x160x1x1x!qElemType1, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV]]
    // CHECK-SAME:      [0, 1, 0, 0] [1, 1, 1, 1] : tensor<1x16x1x1xf16, {order = #NHWC}> to tensor<1x1x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[SLICE]] : tensor<1x1x1x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @OptimizeInnermostSliceMultiple
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x256x2048x4xf16>)
func.func @OptimizeInnermostSliceMultiple(%arg0: tensor<1x256x2048x4xf16>) -> (tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 256, 2048, 1] : tensor<1x256x2048x4xf16> to tensor<1x256x2048x1xf16>
    %1 = IE.Slice %arg0 [0, 0, 0, 1] [1, 256, 2048, 1] : tensor<1x256x2048x4xf16> to tensor<1x256x2048x1xf16>

    return %0, %1 : tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16>


    // First slice path (offset [0, 0, 0, 0])
    // CHECK:   [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x256x2048x4xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_IN_0:%.+]] = IE.ShapeCast {shape = [1, 64, 256, 128]} inputs([[PERMUTE_CAST_0]] : tensor<1x4x256x2048xf16, {order = #NHWC}>) -> tensor<1x64x256x128xf16, {order = #NHWC}>
    // CHECK:   [[WEIGHTS_1:%.+]] = const.Declare tensor<16x64x1x1xf16, {order = #NHWC}>
    // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[SHAPE_CAST_IN_0]], [[WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x256x128xf16, {order = #NHWC}>, tensor<16x64x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_OUT_0:%.+]] = IE.ShapeCast {shape = [1, 1, 256, 2048]} inputs([[CONV_0]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x1x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST_OUT_0:%.+]] = IE.PermuteCast([[SHAPE_CAST_OUT_0]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x256x2048x1xf16>

    // Second slice path (offset [0, 0, 0, 1])
    // CHECK:   [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x256x2048x4xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_IN_1:%.+]] = IE.ShapeCast {shape = [1, 64, 256, 128]} inputs([[PERMUTE_CAST_1]] : tensor<1x4x256x2048xf16, {order = #NHWC}>) -> tensor<1x64x256x128xf16, {order = #NHWC}>
    // CHECK:   [[WEIGHTS_0:%.+]] = const.Declare tensor<16x64x1x1xf16, {order = #NHWC}>
    // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[SHAPE_CAST_IN_1]], [[WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x256x128xf16, {order = #NHWC}>, tensor<16x64x1x1xf16, {order = #NHWC}> -> tensor<1x16x256x128xf16, {order = #NHWC}>
    // CHECK:   [[SHAPE_CAST_OUT_1:%.+]] = IE.ShapeCast {shape = [1, 1, 256, 2048]} inputs([[CONV_1]] : tensor<1x16x256x128xf16, {order = #NHWC}>) -> tensor<1x1x256x2048xf16, {order = #NHWC}>
    // CHECK:   [[PERMUTE_CAST_OUT_1:%.+]] = IE.PermuteCast([[SHAPE_CAST_OUT_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x256x2048x1xf16>

    // CHECK:   return [[PERMUTE_CAST_OUT_0]], [[PERMUTE_CAST_OUT_1]] : tensor<1x256x2048x1xf16>, tensor<1x256x2048x1xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotFuseForCompressConv
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x4x320000x4xf16, {order = #NHWC}>
func.func @NotFuseForCompressConv(%arg0: tensor<1x4x320000x4xf16, {order = #NHWC}>) -> tensor<1x1x320000x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x4x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x320000x4xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}> -> tensor<1x16x320000x4xf16, {order = #NHWC}>
    %1 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 320000, 4] : tensor<1x16x320000x4xf16, {order = #NHWC}> to tensor<1x1x320000x4xf16, {order = #NHWC}>

    return %1 : tensor<1x1x320000x4xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = IE.Convolution
    // CHECK:       [[SLICE:%.+]] = IE.Slice
    // CHECK:       return [[SLICE]]
}
