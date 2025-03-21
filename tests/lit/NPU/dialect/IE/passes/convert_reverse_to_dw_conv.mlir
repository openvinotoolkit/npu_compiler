//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-reverse-to-dw-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @ConvertReverseToGroupConvolutionWithAxesOfHW
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x512x3x3xf16>
func.func @ConvertReverseToGroupConvolutionWithAxesOfHW(%arg0: tensor<512x512x3x3xf16>) -> (tensor<512x512x3x3xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [2, 3],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    return %0 : tensor<512x512x3x3xf16>

    // CHECK-NOT:    IE.Reverse

    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_3:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_4:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_5:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_6:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_7:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_8:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_9:%.+]] = const.Declare tensor<512x1x3x3xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 512, 1536, 3]} inputs([[INPUT]] : tensor<512x512x3x3xf16>) -> tensor<1x512x1536x3xf16>

    // CHECK:        [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_9]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_2:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_8]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_3:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_7]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_4:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_6]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_5:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_5]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_6:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_4]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_7:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_3]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_8:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_9:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>

    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[GROUP_CONV_1]], [[GROUP_CONV_2]], [[GROUP_CONV_3]], [[GROUP_CONV_4]], [[GROUP_CONV_5]], [[GROUP_CONV_6]], [[GROUP_CONV_7]], [[GROUP_CONV_8]], [[GROUP_CONV_9]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16> -> tensor<9x512x512x1xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[CONCAT]]) {order_value = #map} : tensor<9x512x512x1xf16> -> tensor<1x512x512x9xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [512, 512, 3, 3]} inputs([[TRANSPOSE]] : tensor<1x512x512x9xf16>) -> tensor<512x512x3x3xf16>

    // CHECK:        return [[SHAPECAST_OUT]] : tensor<512x512x3x3xf16>
}

// -----

// CHECK:  #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK:  #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @ConvertReverseToGroupConvolutionWithAxesOfNonHW
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x3x3x512xf16>
func.func @ConvertReverseToGroupConvolutionWithAxesOfNonHW(%arg0: tensor<512x3x3x512xf16>) -> (tensor<512x3x3x512xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [1, 2],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x3x3x512xf16> -> tensor<512x3x3x512xf16>

    return %0 : tensor<512x3x3x512xf16>

    // CHECK-NOT:    IE.Reverse

    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_3:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_4:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_5:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_6:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_7:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_8:%.+]] = const.Declare tensor<512x1x3x3xf16>
    // CHECK-DAG:    [[WEIGHTS_9:%.+]] = const.Declare tensor<512x1x3x3xf16>

    // CHECK:        [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<512x3x3x512xf16> -> tensor<512x512x3x3xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 512, 1536, 3]} inputs([[TRANSPOSE_IN]] : tensor<512x512x3x3xf16>) -> tensor<1x512x1536x3xf16>

    // CHECK:        [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_9]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_2:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_8]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_3:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_7]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_4:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_6]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_5:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_5]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_6:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_4]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_7:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_3]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_8:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>
    // CHECK:        [[GROUP_CONV_9:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [3, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x3x3xf16> -> tensor<1x512x512x1xf16>

    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[GROUP_CONV_1]], [[GROUP_CONV_2]], [[GROUP_CONV_3]], [[GROUP_CONV_4]], [[GROUP_CONV_5]], [[GROUP_CONV_6]], [[GROUP_CONV_7]], [[GROUP_CONV_8]], [[GROUP_CONV_9]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16>, tensor<1x512x512x1xf16> -> tensor<9x512x512x1xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[CONCAT]]) {order_value = #map} : tensor<9x512x512x1xf16> -> tensor<1x512x512x9xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [512, 512, 3, 3]} inputs([[TRANSPOSE]] : tensor<1x512x512x9xf16>) -> tensor<512x512x3x3xf16>
    // CHECK:        [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[SHAPECAST_OUT]]) {order_value = #NHWC} : tensor<512x512x3x3xf16> -> tensor<512x3x3x512xf16>

    // CHECK:        return [[TRANSPOSE_OUT]] : tensor<512x3x3x512xf16>
}

// -----

// CHECK-LABEL: @NotConvertReverseToGroupConvolutionWithDiscontinuousAxes
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x512x3x3xf16>
func.func @NotConvertReverseToGroupConvolutionWithDiscontinuousAxes(%arg0: tensor<512x512x3x3xf16>) -> (tensor<512x512x3x3xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [1, 3],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    return %0 : tensor<512x512x3x3xf16>

    // CHECK:        [[REVERSE:%.+]] = IE.Reverse([[INPUT]]) {axis_value = [1, 3], mode = #IE.reverse_mode<INDEX>} : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    // CHECK:        return [[REVERSE]] : tensor<512x512x3x3xf16>
}

// -----

// CHECK-LABEL: @NotConvertReverseToGroupConvolutionWithLargeAxes
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x512x9x9xf16>
func.func @NotConvertReverseToGroupConvolutionWithLargeAxes(%arg0: tensor<512x512x9x9xf16>) -> (tensor<512x512x9x9xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [2, 3],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x512x9x9xf16> -> tensor<512x512x9x9xf16>

    return %0 : tensor<512x512x9x9xf16>

    // CHECK:        [[REVERSE:%.+]] = IE.Reverse([[INPUT]]) {axis_value = [2, 3], mode = #IE.reverse_mode<INDEX>} : tensor<512x512x9x9xf16> -> tensor<512x512x9x9xf16>

    // CHECK:        return [[REVERSE]] : tensor<512x512x9x9xf16>
}

// -----

// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @ConvertReverseToGroupConvolutionWithAxisOfW
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x512x3x3xf16>
func.func @ConvertReverseToGroupConvolutionWithAxisOfW(%arg0: tensor<512x512x3x3xf16>) -> (tensor<512x512x3x3xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [3],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    return %0 : tensor<512x512x3x3xf16>

    // CHECK-NOT:    IE.Reverse

    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<512x1x1x3xf16>
    // CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<512x1x1x3xf16>
    // CHECK-DAG:    [[WEIGHTS_3:%.+]] = const.Declare tensor<512x1x1x3xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 512, 1536, 3]} inputs([[INPUT]] : tensor<512x512x3x3xf16>) -> tensor<1x512x1536x3xf16>

    // CHECK:        [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_3]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>
    // CHECK:        [[GROUP_CONV_2:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>
    // CHECK:        [[GROUP_CONV_3:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>

    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[GROUP_CONV_1]], [[GROUP_CONV_2]], [[GROUP_CONV_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x512x1536x1xf16>, tensor<1x512x1536x1xf16>, tensor<1x512x1536x1xf16> -> tensor<3x512x1536x1xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[CONCAT]]) {order_value = #map} : tensor<3x512x1536x1xf16> -> tensor<1x512x1536x3xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [512, 512, 3, 3]} inputs([[TRANSPOSE]] : tensor<1x512x1536x3xf16>) -> tensor<512x512x3x3xf16>

    // CHECK:        return [[SHAPECAST_OUT]] : tensor<512x512x3x3xf16>
}

// -----

// CHECK:  #NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
// CHECK:  #map = affine_map<(d0, d1, d2, d3) -> (d3, d1, d2, d0)>

// CHECK-LABEL: @ConvertReverseToGroupConvolutionWithAxisOfNonW
// CHECK-SAME:    [[INPUT:%.+]]: tensor<512x512x3x3xf16>
func.func @ConvertReverseToGroupConvolutionWithAxisOfNonW(%arg0: tensor<512x512x3x3xf16>) -> (tensor<512x512x3x3xf16>) {
    %0 = IE.Reverse(%arg0) {
        axis_value = [2],
        mode = #IE.reverse_mode<INDEX>
    } : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    return %0 : tensor<512x512x3x3xf16>

    // CHECK-NOT:    IE.Reverse

    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<512x1x1x3xf16>
    // CHECK-DAG:    [[WEIGHTS_2:%.+]] = const.Declare tensor<512x1x1x3xf16>
    // CHECK-DAG:    [[WEIGHTS_3:%.+]] = const.Declare tensor<512x1x1x3xf16>

    // CHECK:        [[TRANSPOSE_IN:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NCWH} : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = IE.ShapeCast {shape = [1, 512, 1536, 3]} inputs([[TRANSPOSE_IN]] : tensor<512x512x3x3xf16>) -> tensor<1x512x1536x3xf16>

    // CHECK:        [[GROUP_CONV_1:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_3]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>
    // CHECK:        [[GROUP_CONV_2:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>
    // CHECK:        [[GROUP_CONV_3:%.+]] = IE.GroupConvolution([[SHAPECAST_IN]], [[WEIGHTS_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1536x3xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1536x1xf16>

    // CHECK:        [[CONCAT:%.+]] = IE.Concat([[GROUP_CONV_1]], [[GROUP_CONV_2]], [[GROUP_CONV_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x512x1536x1xf16>, tensor<1x512x1536x1xf16>, tensor<1x512x1536x1xf16> -> tensor<3x512x1536x1xf16>
    // CHECK:        [[TRANSPOSE:%.+]] = IE.Transpose([[CONCAT]]) {order_value = #map} : tensor<3x512x1536x1xf16> -> tensor<1x512x1536x3xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = IE.ShapeCast {shape = [512, 512, 3, 3]} inputs([[TRANSPOSE]] : tensor<1x512x1536x3xf16>) -> tensor<512x512x3x3xf16>
    // CHECK:        [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[SHAPECAST_OUT]]) {order_value = #NCWH} : tensor<512x512x3x3xf16> -> tensor<512x512x3x3xf16>

    // CHECK:        return [[TRANSPOSE_OUT]] : tensor<512x512x3x3xf16>
}
