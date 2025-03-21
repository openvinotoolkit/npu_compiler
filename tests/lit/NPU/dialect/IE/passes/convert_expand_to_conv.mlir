//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-expand-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv16Channels
func.func @ConvertExpandToConv16Channels(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 224]
    // CHECK-SAME:  } : tensor<1x256x64x14xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv4Channels
func.func @ConvertExpandToConv4Channels(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x4x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x4x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x4x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000000000000000{{([0]{168})}}
    // CHECK-SAME:      0000003C0000000000000000{{([0]{168})}}
    // CHECK-SAME:      00000000003C000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000003C00000000{{([0]{168})}}
    // CHECK-SAME:      0000000000000000003C0000{{([0]{168})}}
    // CHECK-SAME:      00000000000000000000003C{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x64x14xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 64, 224]
    // CHECK-SAME:  } : tensor<1x64x64x14xf16, {order = #NHWC}> -> tensor<1x4x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv16ChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x64x220xf16, {order = #NHWC}>)
func.func @ConvertExpandToConv16ChannelsPadded(%arg0: tensor<1x3x64x220xf16, {order = #NHWC}>)
    -> tensor<1x16x64x220xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x220xf16, {order = #NHWC}> -> tensor<1x16x64x220xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x220xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 42240, 1]
    // CHECK-SAME:  } : tensor<1x3x64x220xf16, {order = #NHWC}> -> tensor<1x1x42240x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 768, 0]
    // CHECK-SAME:  } : tensor<1x1x42240x1xf16, {order = #NHWC}> -> tensor<1x1x43008x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x1x43008x1xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 229376, 1]
    // CHECK-SAME:  } : tensor<1x256x64x14xf16, {order = #NHWC}> -> tensor<1x1x229376x1xf16, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 225280, 1] : tensor<1x1x229376x1xf16, {order = #NHWC}> to tensor<1x1x225280x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 220]
    // CHECK-SAME:  } : tensor<1x1x225280x1xf16, {order = #NHWC}> -> tensor<1x16x64x220xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x220xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConv4ChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x64x220xf16, {order = #NHWC}>)
func.func @ConvertExpandToConv4ChannelsPadded(%arg0: tensor<1x3x64x220xf16, {order = #NHWC}>)
    -> tensor<1x4x64x220xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x64x220xf16, {order = #NHWC}> -> tensor<1x4x64x220xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x4x64x220xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000000000000000{{([0]{168})}}
    // CHECK-SAME:      0000003C0000000000000000{{([0]{168})}}
    // CHECK-SAME:      00000000003C000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}
    // CHECK-SAME:      000000000000003C00000000{{([0]{168})}}
    // CHECK-SAME:      0000000000000000003C0000{{([0]{168})}}
    // CHECK-SAME:      00000000000000000000003C{{([0]{168})}}
    // CHECK-SAME:      000000000000000000000000{{([0]{168})}}

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 42240, 1]
    // CHECK-SAME:  } : tensor<1x3x64x220xf16, {order = #NHWC}> -> tensor<1x1x42240x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 768, 0]
    // CHECK-SAME:  } : tensor<1x1x42240x1xf16, {order = #NHWC}> -> tensor<1x1x43008x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x1x43008x1xf16, {order = #NHWC}> -> tensor<1x48x64x14xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x64x14xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 57344, 1]
    // CHECK-SAME:  } : tensor<1x64x64x14xf16, {order = #NHWC}> -> tensor<1x1x57344x1xf16, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 56320, 1] : tensor<1x1x57344x1xf16, {order = #NHWC}> to tensor<1x1x56320x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 64, 220]
    // CHECK-SAME:  } : tensor<1x1x56320x1xf16, {order = #NHWC}> -> tensor<1x4x64x220xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x64x220xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0078657811763239837:128>

// CHECK-LABEL: @ConvertExpandAvgPoolToConv16ChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x513x513xf16, {order = #NHWC}>)
func.func @ConvertExpandAvgPoolToConv16ChannelsPadded(%arg0: tensor<1x3x513x513xf16, {order = #NHWC}>)
    -> tensor<1x16x513x513x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513xf16, {order = #NHWC}>

    %AVGPOOL = IE.AvgPool(%EXPAND) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513x!qElemType, {order = #NHWC}>

    return %AVGPOOL : tensor<1x16x513x513x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 789507, 1]
    // CHECK-SAME:  } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x1x789507x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 23085, 0]
    // CHECK-SAME:  } : tensor<1x1x789507x1xf16, {order = #NHWC}> -> tensor<1x1x812592x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 513, 33]
    // CHECK-SAME:  } : tensor<1x1x812592x1xf16, {order = #NHWC}> -> tensor<1x48x513x33xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x513x33xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x513x33x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 4333824, 1]
    // CHECK-SAME:  } : tensor<1x256x513x33x!qElemType, {order = #NHWC}> -> tensor<1x1x4333824x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 4210704, 1] : tensor<1x1x4333824x1x!qElemType, {order = #NHWC}> to tensor<1x1x4210704x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 513, 513]
    // CHECK-SAME:  } : tensor<1x1x4210704x1x!qElemType, {order = #NHWC}> -> tensor<1x16x513x513x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x513x513x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0078657811763239837:128>

// CHECK-LABEL: @ConvertExpandAvgPoolToConv4ChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x513x513xf16, {order = #NHWC}>)
func.func @ConvertExpandAvgPoolToConv4ChannelsPadded(%arg0: tensor<1x3x513x513xf16, {order = #NHWC}>)
    -> tensor<1x3x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513xf16, {order = #NHWC}>

    %AVGPOOL = IE.AvgPool(%EXPAND) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    %SLICE = IE.Slice %AVGPOOL [0, 0, 0, 0] [1, 3, 513, 513] : tensor<1x16x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}> to tensor<1x3x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    return %SLICE : tensor<1x3x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 789507, 1]
    // CHECK-SAME:  } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x1x789507x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 23085, 0]
    // CHECK-SAME:  } : tensor<1x1x789507x1xf16, {order = #NHWC}> -> tensor<1x1x812592x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 513, 33]
    // CHECK-SAME:  } : tensor<1x1x812592x1xf16, {order = #NHWC}> -> tensor<1x48x513x33xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x513x33xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x513x33x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 1083456, 1]
    // CHECK-SAME:  } : tensor<1x64x513x33x!qElemType, {order = #NHWC}> -> tensor<1x1x1083456x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 1052676, 1] : tensor<1x1x1083456x1x!qElemType, {order = #NHWC}> to tensor<1x1x1052676x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 513, 513]
    // CHECK-SAME:  } : tensor<1x1x1052676x1x!qElemType, {order = #NHWC}> -> tensor<1x4x513x513x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE_OUT:%.+]] = IE.Slice [[RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 3, 513, 513] : tensor<1x4x513x513x!qElemType, {order = #NHWC}> to tensor<1x3x513x513x!qElemType, {order = #NHWC}>

    // CHECK:   return [[SLICE_OUT]] : tensor<1x3x513x513x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0078657811763239837:128>

// CHECK-LABEL: @ConvertExpandAvgPoolToConv4ChannelsPaddedAndFold
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x513x513xf16, {order = #NHWC}>)
func.func @ConvertExpandAvgPoolToConv4ChannelsPaddedAndFold(%arg0: tensor<1x3x513x513xf16, {order = #NHWC}>)
    -> tensor<1x4x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513xf16, {order = #NHWC}>

    %AVGPOOL = IE.AvgPool(%EXPAND) {
        exclude_pads,
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x513x513xf16, {order = #NHWC}> -> tensor<1x16x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    %SLICE = IE.Slice %AVGPOOL [0, 0, 0, 0] [1, 4, 513, 513] : tensor<1x16x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}> to tensor<1x4x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    return %SLICE : tensor<1x4x513x513x!quant.uniform<u8:f16, 0.0078657811763239837:128>, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 789507, 1]
    // CHECK-SAME:  } : tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x1x789507x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 23085, 0]
    // CHECK-SAME:  } : tensor<1x1x789507x1xf16, {order = #NHWC}> -> tensor<1x1x812592x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 513, 33]
    // CHECK-SAME:  } : tensor<1x1x812592x1xf16, {order = #NHWC}> -> tensor<1x48x513x33xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x513x33xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x513x33x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 1083456, 1]
    // CHECK-SAME:  } : tensor<1x64x513x33x!qElemType, {order = #NHWC}> -> tensor<1x1x1083456x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 1052676, 1] : tensor<1x1x1083456x1x!qElemType, {order = #NHWC}> to tensor<1x1x1052676x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 513, 513]
    // CHECK-SAME:  } : tensor<1x1x1052676x1x!qElemType, {order = #NHWC}> -> tensor<1x4x513x513x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x513x513x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemTypeAdd = !quant.uniform<u8:f16, 0.01567845251045975:128>
!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>

// CHECK-LABEL: @ConvertExpandQuantizeToConv4ChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x299x299xf16, {order = #NHWC}>)
func.func @ConvertExpandQuantizeToConv4ChannelsPadded(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
    -> tensor<1x3x299x299x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    %ADD = IE.Add(%EXPAND, %EXPAND) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 3, 299, 299] : tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}> to tensor<1x3x299x299x!qElemTypeAdd, {order = #NHWC}>
    %QUANTCAST = IE.QuantizeCast(%SLICE) {dstElemType = !qElemType} : tensor<1x3x299x299x!qElemTypeAdd, {order = #NHWC}> -> tensor<1x3x299x299x!qElemType, {order = #NHWC}>
    return %QUANTCAST : tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<48x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 268203, 1]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 4485, 0]
    // CHECK-SAME:  } : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 299, 19]
    // CHECK-SAME:  } : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<48x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x48x299x19x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 272688, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19x!qElemType, {order = #NHWC}> -> tensor<1x1x272688x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 268203, 1] : tensor<1x1x272688x1x!qElemType, {order = #NHWC}> to tensor<1x1x268203x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 3, 299, 299]
    // CHECK-SAME:  } : tensor<1x1x268203x1x!qElemType, {order = #NHWC}> -> tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x3x299x299x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemTypeAdd = !quant.uniform<u8:f16, 0.01567845251045975:128>
!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>

// CHECK-LABEL: @ConvertExpandQuantize9OutChannelsPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x299x299xf16, {order = #NHWC}>)
func.func @ConvertExpandQuantize9OutChannelsPadded(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
    -> tensor<1x9x299x299x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    %ADD = IE.Add(%EXPAND, %EXPAND) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 9, 299, 299] : tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}> to tensor<1x9x299x299x!qElemTypeAdd, {order = #NHWC}>
    %QUANTCAST = IE.QuantizeCast(%SLICE) {dstElemType = !qElemType} : tensor<1x9x299x299x!qElemTypeAdd, {order = #NHWC}> -> tensor<1x9x299x299x!qElemType, {order = #NHWC}>
    return %QUANTCAST : tensor<1x9x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<144x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 268203, 1]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 4485, 0]
    // CHECK-SAME:  } : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 299, 19]
    // CHECK-SAME:  } : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<144x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x144x299x19x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 818064, 1]
    // CHECK-SAME:  } : tensor<1x144x299x19x!qElemType, {order = #NHWC}> -> tensor<1x1x818064x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 804609, 1] : tensor<1x1x818064x1x!qElemType, {order = #NHWC}> to tensor<1x1x804609x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 9, 299, 299]
    // CHECK-SAME:  } : tensor<1x1x804609x1x!qElemType, {order = #NHWC}> -> tensor<1x9x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x9x299x299x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemTypeAdd = !quant.uniform<u8:f16, 0.01567845251045975:128>
!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>

// CHECK-LABEL: @ConvertExpandQuantizeSlicePadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x299x299xf16, {order = #NHWC}>)
func.func @ConvertExpandQuantizeSlicePadded(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
    -> tensor<1x3x299x299x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    %ADD = IE.Add(%EXPAND, %EXPAND) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 9, 299, 299] : tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}> to tensor<1x9x299x299x!qElemTypeAdd, {order = #NHWC}>
    %QUANTCAST = IE.QuantizeCast(%SLICE) {dstElemType = !qElemType} : tensor<1x9x299x299x!qElemTypeAdd, {order = #NHWC}> -> tensor<1x9x299x299x!qElemType, {order = #NHWC}>
    %SLICE2 = IE.Slice %QUANTCAST [0, 0, 0, 0] [1, 3, 299, 299] : tensor<1x9x299x299x!qElemType, {order = #NHWC}> to tensor<1x3x299x299x!qElemType, {order = #NHWC}>
    return %SLICE2 : tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 268203, 1]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 4485, 0]
    // CHECK-SAME:  } : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 299, 19]
    // CHECK-SAME:  } : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x299x19x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 363584, 1]
    // CHECK-SAME:  } : tensor<1x64x299x19x!qElemType, {order = #NHWC}> -> tensor<1x1x363584x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 357604, 1] : tensor<1x1x363584x1x!qElemType, {order = #NHWC}> to tensor<1x1x357604x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 299, 299]
    // CHECK-SAME:  } : tensor<1x1x357604x1x!qElemType, {order = #NHWC}> -> tensor<1x4x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 3, 299, 299] : tensor<1x4x299x299x!qElemType, {order = #NHWC}> to tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[SLICE2]] : tensor<1x3x299x299x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemTypeAdd = !quant.uniform<u8:f16, 0.01567845251045975:128>
!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.01567845251045975:128>

// CHECK-LABEL: @ConvertExpandSkipSliceQuantizeOnHWPadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x299x299xf16, {order = #NHWC}>)
func.func @ConvertExpandSkipSliceQuantizeOnHWPadded(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
    -> tensor<1x9x200x200x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    %ADD = IE.Add(%EXPAND, %EXPAND) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}>
    %SLICE = IE.Slice %ADD [0, 0, 0, 0] [1, 9, 200, 200] : tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}> to tensor<1x9x200x200x!qElemTypeAdd, {order = #NHWC}>
    %QUANTCAST = IE.QuantizeCast(%SLICE) {dstElemType = !qElemType} : tensor<1x9x200x200x!qElemTypeAdd, {order = #NHWC}> -> tensor<1x9x200x200x!qElemType, {order = #NHWC}>
    return %QUANTCAST : tensor<1x9x200x200x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 268203, 1]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 4485, 0]
    // CHECK-SAME:  } : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 299, 19]
    // CHECK-SAME:  } : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x299x19xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 1454336, 1]
    // CHECK-SAME:  } : tensor<1x256x299x19xf16, {order = #NHWC}> -> tensor<1x1x1454336x1xf16, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 1430416, 1] : tensor<1x1x1454336x1xf16, {order = #NHWC}> to tensor<1x1x1430416x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 299, 299]
    // CHECK-SAME:  } : tensor<1x1x1430416x1xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.+]] = IE.Add([[RESHAPE_OUTPUT]], [[RESHAPE_OUTPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemType1, {order = #NHWC}>

    // CHECK:   [[SLICE2:%.+]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 9, 200, 200] : tensor<1x16x299x299x!qElemType1, {order = #NHWC}> to tensor<1x9x200x200x!qElemType1, {order = #NHWC}>

    // CHECK:   [[QUANTCAST:%.+]] = IE.QuantizeCast([[SLICE2]]) {dstElemType = !qElemType} : tensor<1x9x200x200x!qElemType1, {order = #NHWC}> -> tensor<1x9x200x200x!qElemType, {order = #NHWC}>

    // CHECK:   return [[QUANTCAST]] : tensor<1x9x200x200x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemTypeAdd = !quant.uniform<u8:f16, 0.01567845251045975:128>
!qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>
// CHECK: !qElemType = !quant.uniform<u8:f16, 0.0078392262552298749:128>

// CHECK-LABEL: @ConvertExpandQuantizeNoSlicePadded
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x299x299xf16, {order = #NHWC}>)
func.func @ConvertExpandQuantizeNoSlicePadded(%arg0: tensor<1x3x299x299xf16, {order = #NHWC}>)
    -> tensor<1x16x299x299x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299xf16, {order = #NHWC}>

    %ADD = IE.Add(%EXPAND, %EXPAND) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x16x299x299xf16, {order = #NHWC}>, tensor<1x16x299x299xf16, {order = #NHWC}> -> tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}>
    %QUANTCAST = IE.QuantizeCast(%ADD) {dstElemType = !qElemType} : tensor<1x16x299x299x!qElemTypeAdd, {order = #NHWC}> -> tensor<1x16x299x299x!qElemType, {order = #NHWC}>
    return %QUANTCAST : tensor<1x16x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_INPUT:%.+]] = IE.AffineReshape([[ARG0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 268203, 1]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16, {order = #NHWC}> -> tensor<1x1x268203x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_INPUT:%.+]] = IE.Expand([[LINEAR_RESHAPE_INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 4485, 0]
    // CHECK-SAME:  } : tensor<1x1x268203x1xf16, {order = #NHWC}> -> tensor<1x1x272688x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND_INPUT]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 299, 19]
    // CHECK-SAME:  } : tensor<1x1x272688x1xf16, {order = #NHWC}> -> tensor<1x48x299x19xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x299x19xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x299x19x!qElemType, {order = #NHWC}>

    // CHECK:   [[LINEAR_RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 1, 1454336, 1]
    // CHECK-SAME:  } : tensor<1x256x299x19x!qElemType, {order = #NHWC}> -> tensor<1x1x1454336x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LINEAR_RESHAPE_OUTPUT]] [0, 0, 0, 0] [1, 1, 1430416, 1] : tensor<1x1x1454336x1x!qElemType, {order = #NHWC}> to tensor<1x1x1430416x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[SLICE]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 299, 299]
    // CHECK-SAME:  } : tensor<1x1x1430416x1x!qElemType, {order = #NHWC}> -> tensor<1x16x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x299x299x!qElemType, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SkipExpandNCHW
func.func @SkipExpandNCHW(%arg0: tensor<1x3x64x224xf16>) -> tensor<1x16x64x224xf16> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf16> -> tensor<1x16x64x224xf16>

    return %EXPAND : tensor<1x16x64x224xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16> -> tensor<1x16x64x224xf16>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf16>
}

// -----

// CHECK-LABEL: @SkipExpand3d
func.func @SkipExpand3d(%arg0: tensor<1x3x64xf16>) -> tensor<1x16x64xf16> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0],
        pads_end = [0, 13, 0]
    } : tensor<1x3x64xf16> -> tensor<1x16x64xf16>

    return %EXPAND : tensor<1x16x64xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0]
    // CHECK-SAME:  } : tensor<1x3x64xf16> -> tensor<1x16x64xf16>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipNonZeroPadsBegin
func.func @SkipNonZeroPadsBegin(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 1, 0, 0],
        pads_end = [0, 12, 0, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 1, 0, 0],
    // CHECK-SAME:      pads_end = [0, 12, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipPaddingOverHeight
func.func @SkipPaddingOverHeight(%arg0: tensor<1x3x64x224xf16, {order = #NHWC}>)
    -> tensor<1x3x66x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 2, 0]
    } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x3x66x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x3x66x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 2, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf16, {order = #NHWC}> -> tensor<1x3x66x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x3x66x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipBatch
func.func @SkipBatch(%arg0: tensor<2x3x64x224xf16, {order = #NHWC}>)
    -> tensor<2x16x64x224xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<2x3x64x224xf16, {order = #NHWC}> -> tensor<2x16x64x224xf16, {order = #NHWC}>

    return %EXPAND : tensor<2x16x64x224xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<2x3x64x224xf16, {order = #NHWC}> -> tensor<2x16x64x224xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<2x16x64x224xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipFloat32Expand
func.func @SkipFloat32Expand(%arg0: tensor<1x3x64x224xf32, {order = #NHWC}>)
    -> tensor<1x16x64x224xf32, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224xf32, {order = #NHWC}> -> tensor<1x16x64x224xf32, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224xf32, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x64x224xf32, {order = #NHWC}> -> tensor<1x16x64x224xf32, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x16x64x224xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>

// CHECK-DAG: [[Q_ACT:!.*]] = !quant.uniform<u8:f16, 2.000000e+00>
// CHECK-DAG: [[Q_WEIGHTS:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT and Q_WEIGHTS
func.func @ConvertQuantizedExpand(%arg0: tensor<1x3x64x224x!qElemType, {order = #NHWC}>)
    -> tensor<1x16x64x224x!qElemType, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x64x224x!qElemType, {order = #NHWC}> -> tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x16x64x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1x[[Q_WEIGHTS]], {order = #NHWC}> = dense<"0x
    // CHECK-SAME:      003C00000000{{([0]{180})}}
    // CHECK-SAME:      0000003C0000{{([0]{180})}}
    // CHECK-SAME:      00000000003C{{([0]{180})}}
    // CHECK-SAME:      000000000000{{([0]{180})}}

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 64, 14]
    // CHECK-SAME:  } : tensor<1x3x64x224x[[Q_ACT]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x64x14x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x64x14x[[Q_ACT]], {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1x[[Q_WEIGHTS]], {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x64x14x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 64, 224]
    // CHECK-SAME:  } : tensor<1x256x64x14x[[Q_ACT]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x64x224x[[Q_ACT]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x64x224x[[Q_ACT]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertLargeExpand
func.func @ConvertLargeExpand(%arg0: tensor<1x3x512x896xf16, {order = #NHWC}>)
    -> tensor<1x16x512x896xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x512x896xf16, {order = #NHWC}> -> tensor<1x16x512x896xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x16x512x896xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Expand
    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 48, 512, 56]
    // CHECK-SAME:  } : tensor<1x3x512x896xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x48x512x56xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x48x512x56xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x48x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x512x56xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 512, 896]
    // CHECK-SAME:  } : tensor<1x256x512x56xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x512x896xf16, {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x16x512x896xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_TYPE
func.func @FuseQuantizeProducer(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 5]
    } inputs(%arg0 : tensor<1x1x19x80xf16, {order = #NHWC}>) -> tensor<1x16x19x5xf16, {order = #NHWC}>

    %ADD = IE.Add(%IN_SHAPE_CAST, %IN_SHAPE_CAST) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x16x19x5xf16, {order = #NHWC}>,
        tensor<1x16x19x5xf16, {order = #NHWC}>
            -> tensor<1x16x19x5x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 1, 19, 80]
    } inputs(%ADD : tensor<1x16x19x5x!qElemType, {order = #NHWC}>) -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %QUANT_CAST = IE.QuantizeCast(%OUT_SHAPE_CAST) {
        dstElemType = !qElemType1
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType1, {order = #NHWC}>

    %EXPAND = IE.Expand(%QUANT_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType1, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 2.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_ACT_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT_TYPE
func.func @FuseQuantizeWithoutShapeCast(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}> {
    %ADD = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>
    } : tensor<1x1x19x80xf16, {order = #NHWC}>,
        tensor<1x1x19x80xf16, {order = #NHWC}>
            -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %QUANT_CAST = IE.QuantizeCast(%ADD) {
        dstElemType = !qElemType1
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType1, {order = #NHWC}>

    %EXPAND = IE.Expand(%QUANT_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType1, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType1, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_ACT_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_ACT_TYPE
func.func @FuseQuantizeWithAvgPool(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType, {order = #NHWC}> {
    %AvgPool = IE.AvgPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x1x19x80xf16, {order = #NHWC}> -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %EXPAND = IE.Expand(%AvgPool) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_ACT_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_ACT_TYPE]], {order = #NHWC}>
}



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_TYPE
func.func @FuseQuantizeWithShapeCastAvgPool(%arg0: tensor<1x1x19x80xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 16, 19, 5]
    } inputs(%arg0 : tensor<1x1x19x80xf16, {order = #NHWC}>) -> tensor<1x16x19x5xf16, {order = #NHWC}>

    %AvgPool = IE.AvgPool(%IN_SHAPE_CAST) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x16x19x5xf16, {order = #NHWC}> -> tensor<1x16x19x5x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 1, 19, 80]
    } inputs(%AvgPool : tensor<1x16x19x5x!qElemType, {order = #NHWC}>) -> tensor<1x1x19x80x!qElemType, {order = #NHWC}>

    %EXPAND = IE.Expand(%OUT_SHAPE_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 3, 0, 0]
    } : tensor<1x1x19x80x!qElemType, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 5]
    // CHECK-SAME:  } : tensor<1x1x19x80xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x5xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x5xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<64x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x64x19x5x[[Q_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<u8:f16, 1.000000e+00>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets Q_TYPE
func.func @FuseQuantizeWithShapeCastAvgPoolDifferentShapes(%arg0: tensor<1x1x19x160xf16, {order = #NHWC}>)
    -> tensor<1x4x19x80x!qElemType, {order = #NHWC}> {
    %IN_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 32, 19, 5]
    } inputs(%arg0 : tensor<1x1x19x160xf16, {order = #NHWC}>) -> tensor<1x32x19x5xf16, {order = #NHWC}>

    %AvgPool = IE.AvgPool(%IN_SHAPE_CAST) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
    } : tensor<1x32x19x5xf16, {order = #NHWC}> -> tensor<1x32x19x5x!qElemType, {order = #NHWC}>

    %OUT_SHAPE_CAST = IE.ShapeCast {
        shape = [1, 2, 19, 80]
    } inputs(%AvgPool : tensor<1x32x19x5x!qElemType, {order = #NHWC}>) -> tensor<1x2x19x80x!qElemType, {order = #NHWC}>

    %EXPAND = IE.Expand(%OUT_SHAPE_CAST) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 2, 0, 0]
    } : tensor<1x2x19x80x!qElemType, {order = #NHWC}> -> tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    return %EXPAND : tensor<1x4x19x80x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 19, 10]
    // CHECK-SAME:  } : tensor<1x1x19x160xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x16x19x10xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x19x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x32x19x10x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.*]] = IE.AffineReshape([[CONV]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 4, 19, 80]
    // CHECK-SAME:  } : tensor<1x32x19x10x[[Q_TYPE]], {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>

    // CHECK:   return [[RESHAPE_OUTPUT]] : tensor<1x4x19x80x[[Q_TYPE]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipLargeKernelAndSmallSpatialSize
func.func @SkipLargeKernelAndSmallSpatialSize(%arg0: tensor<1x387x45x80xf16, {order = #NHWC}>)
    -> tensor<1x400x45x80xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x387x45x80xf16, {order = #NHWC}> -> tensor<1x400x45x80xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x400x45x80xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 13, 0, 0]
    // CHECK-SAME:  } : tensor<1x387x45x80xf16, {order = #NHWC}> -> tensor<1x400x45x80xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x400x45x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertWithBigChannelAndHeightRatio
func.func @NotConvertWithBigChannelAndHeightRatio(%arg0: tensor<1x60x80x80xf16, {order = #NHWC}>)
    -> tensor<1x64x80x80xf16, {order = #NHWC}> {
    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 4, 0, 0]
    } : tensor<1x60x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    return %EXPAND : tensor<1x64x80x80xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand({{[^:]+}}) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 4, 0, 0]
    // CHECK-SAME:  } : tensor<1x60x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // CHECK:   return [[EXPAND]] : tensor<1x64x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertExpandToConvWithNon16MultipleWidth
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x32x79741xf16, {order = #NHWC}>
func.func @ConvertExpandToConvWithNon16MultipleWidth(%arg0: tensor<1x1x32x79741xf16, {order = #NHWC}>)
    -> tensor<1x80x32x15949xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x1x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>]

    %EXPAND = IE.Expand(%arg0) {
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 15, 0, 0]
    } : tensor<1x1x32x79741xf16, {order = #NHWC}> -> tensor<1x16x32x79741xf16, {order = #NHWC}>

    %CONV = IE.Convolution(%EXPAND, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 5]
    } : tensor<1x16x32x79741xf16, {order = #NHWC}>, tensor<80x16x1x1xf16, {order = #NHWC}> -> tensor<1x80x32x15949xf16, {order = #NHWC}>

    return %CONV : tensor<1x80x32x15949xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<256x16x1x1xf16, {order = #NHWC}>
    // CHECK:   [[CST:%.+]] = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[INPUT]]) {
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 3]
    // CHECK-SAME:  } : tensor<1x1x32x79741xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x1x32x79744xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_INPUT:%.+]] = IE.AffineReshape([[EXPAND]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 32, 4984]
    // CHECK-SAME:  } : tensor<1x1x32x79744xf16, {order = #NHWC}> -> tensor<1x16x32x4984xf16, {order = #NHWC}>

    // CHECK:   [[CONV0:%.+]] = IE.Convolution([[RESHAPE_INPUT]], [[EXPAND_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x32x4984xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<256x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x256x32x4984xf16, {order = #NHWC}>

    // CHECK:   [[RESHAPE_OUTPUT:%.+]] = IE.AffineReshape([[CONV0]]) {
    // CHECK-SAME:      dim_mapping = [
    // CHECK-SAME:          [0], [1], [2], [3]
    // CHECK-SAME:      ],
    // CHECK-SAME:      shape_value = [1, 16, 32, 79744]
    // CHECK-SAME:  } : tensor<1x256x32x4984xf16, {order = #NHWC}> -> tensor<1x16x32x79744xf16, {order = #NHWC}>

    // CHECK:   [[CONV1:%.+]] = IE.Convolution([[RESHAPE_OUTPUT]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 5]
    // CHECK-SAME:  } : tensor<1x16x32x79744xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<80x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x80x32x15949xf16, {order = #NHWC}>

    // CHECK:   return [[CONV1]] : tensor<1x80x32x15949xf16, {order = #NHWC}>
}
