//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-conv3d-to-conv2d %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: @ConvertNceOpsTo4DConvolution3DCommonCase
func.func @ConvertNceOpsTo4DConvolution3DCommonCase(%arg0: tensor<1x1x3x56x56xf16>) -> tensor<1x32x3x28x28xf16> {
    %FILTERS = const.Declare tensor<32x1x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1, 1, 1], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x1x3x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x3x28x28xf16>

    return %RESULT : tensor<1x32x3x28x28xf16>

    // CHECK-DAG:   [[CST_WEIGHTS:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x3x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_0]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x3x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_1]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x3x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[RESHAPE_2]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[CONV_0]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[CONV_2]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_3]], [[RESHAPE_4]], [[RESHAPE_5]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x3x784xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 28, 28]} : tensor<1x32x3x784xf16> -> tensor<1x32x3x28x28xf16>

    // CHECK:       return [[RESHAPE_6]]

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution3DCommonCaseNoPad
func.func @ConvertNceOpsTo4DConvolution3DCommonCaseNoPad(%arg0: tensor<1x32x5x28x28xf16>) -> tensor<1x32x3x27x27xf16> {
    %FILTERS = const.Declare tensor<32x32x3x2x2xf16> = dense<1.000000e+00> : tensor<32x32x3x2x2xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], strides = [1, 1, 1]} : tensor<1x32x5x28x28xf16>, tensor<32x32x3x2x2xf16> -> tensor<1x32x3x27x27xf16>

    return %RESULT : tensor<1x32x3x27x27xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_0]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_1]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[RESHAPE_2]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>

    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_3:%.+]] = IE.Convolution([[RESHAPE_3]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_4:%.+]] = IE.Convolution([[RESHAPE_4]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_5:%.+]] = IE.Convolution([[RESHAPE_5]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_3:%.+]] = IE.Add([[CONV_3]], [[CONV_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_4:%.+]] = IE.Add([[ADD_3]], [[CONV_5]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>

    // CHECK:       [[SLICE_6:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_6:%.+]] = IE.Convolution([[RESHAPE_6]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[SLICE_7]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_7:%.+]] = IE.Convolution([[RESHAPE_7]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[SLICE_8:%.+]] = IE.Slice {{[^:]+}} [0, 0, 4, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.Reshape([[SLICE_8]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_8:%.+]] = IE.Convolution([[RESHAPE_8]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_5:%.+]] = IE.Add([[CONV_6]], [[CONV_7]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>
    // CHECK:       [[ADD_6:%.+]] = IE.Add([[ADD_5]], [[CONV_8]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x27x27xf16>, tensor<1x32x27x27xf16> -> tensor<1x32x27x27xf16>

    // CHECK:       [[RESHAPE_9:%.+]] = IE.Reshape([[ADD_2]]) {shape_value = [1, 32, 1, 729]} : tensor<1x32x27x27xf16> -> tensor<1x32x1x729xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[ADD_4]]) {shape_value = [1, 32, 1, 729]} : tensor<1x32x27x27xf16> -> tensor<1x32x1x729xf16>
    // CHECK:       [[RESHAPE_11:%.+]] = IE.Reshape([[ADD_6]]) {shape_value = [1, 32, 1, 729]} : tensor<1x32x27x27xf16> -> tensor<1x32x1x729xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_9]], [[RESHAPE_10]], [[RESHAPE_11]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x729xf16>, tensor<1x32x1x729xf16>, tensor<1x32x1x729xf16> -> tensor<1x32x3x729xf16>
    // CHECK:       [[RESHAPE_12:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 27, 27]} : tensor<1x32x3x729xf16> -> tensor<1x32x3x27x27xf16>

    // CHECK:       return [[RESHAPE_12]]
}


// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution3DOnlyDepth
func.func @ConvertNceOpsTo4DConvolution3DOnlyDepth(%arg0: tensor<1x32x5x28x28xf16>) -> tensor<1x32x5x28x28xf16> {
    %FILTERS = const.Declare tensor<32x32x3x2x2xf16> = dense<1.000000e+00> : tensor<32x32x3x2x2xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1, 1, 1], pads_begin = [1, 1, 1], pads_end = [1, 0, 0], strides = [1, 1, 1]} : tensor<1x32x5x28x28xf16>, tensor<32x32x3x2x2xf16> -> tensor<1x32x5x28x28xf16>

    return %RESULT : tensor<1x32x5x28x28xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_0]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_1]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_3:%.+]] = IE.Convolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_4:%.+]] = IE.Convolution([[RESHAPE_4]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CONV_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[SLICE_5:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_5:%.+]] = IE.Convolution([[RESHAPE_5]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_6:%.+]] = IE.Convolution([[RESHAPE_6]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_7:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[SLICE_7]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_7:%.+]] = IE.Convolution([[RESHAPE_7]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_3:%.+]] = IE.Add([[CONV_5]], [[CONV_6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_4:%.+]] = IE.Add([[ADD_3]], [[CONV_7]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[SLICE_8:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.Reshape([[SLICE_8]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_8:%.+]] = IE.Convolution([[RESHAPE_8]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_9:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.Reshape([[SLICE_9]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_9:%.+]] = IE.Convolution([[RESHAPE_9]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_10:%.+]] = IE.Slice {{[^:]+}} [0, 0, 4, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[SLICE_10]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_10:%.+]] = IE.Convolution([[RESHAPE_10]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_5:%.+]] = IE.Add([[CONV_8]], [[CONV_9]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_6:%.+]] = IE.Add([[ADD_5]], [[CONV_10]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[SLICE_11:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_11:%.+]] = IE.Reshape([[SLICE_11]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_11:%.+]] = IE.Convolution([[RESHAPE_11]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_12:%.+]] = IE.Slice {{[^:]+}} [0, 0, 4, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_12:%.+]] = IE.Reshape([[SLICE_12]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_12:%.+]] = IE.Convolution([[RESHAPE_12]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[ADD_7:%.+]] = IE.Add([[CONV_11]], [[CONV_12]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x28xf16>, tensor<1x32x28x28xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[RESHAPE_13:%.+]] = IE.Reshape([[ADD_0]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_14:%.+]] = IE.Reshape([[ADD_2]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_15:%.+]] = IE.Reshape([[ADD_4]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_16:%.+]] = IE.Reshape([[ADD_6]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_17:%.+]] = IE.Reshape([[ADD_7]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_13]], [[RESHAPE_14]], [[RESHAPE_15]], [[RESHAPE_16]], [[RESHAPE_17]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x5x784xf16>
    // CHECK:       [[RESHAPE_18:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 5, 28, 28]} : tensor<1x32x5x784xf16> -> tensor<1x32x5x28x28xf16>

    // CHECK:       return [[RESHAPE_18]]

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution3DWithStride
func.func @ConvertNceOpsTo4DConvolution3DWithStride(%arg0: tensor<1x32x5x28x28xf16>) -> tensor<1x32x3x28x29xf16> {
    %FILTERS = const.Declare tensor<32x32x3x1x1xf16> = dense<1.000000e+00> : tensor<32x32x3x1x1xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1, 1, 1], pads_begin = [1, 0, 0], pads_end = [1, 0, 1], strides = [2, 1, 1]} : tensor<1x32x5x28x28xf16>, tensor<32x32x3x1x1xf16> -> tensor<1x32x3x28x29xf16>

    return %RESULT : tensor<1x32x3x28x29xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x32x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x32x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x32x1x1xf16> = dense<1.000000e+00> : tensor<32x32x1x1xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_0]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_1]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_3:%.+]] = IE.Convolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_4:%.+]] = IE.Convolution([[RESHAPE_4]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CONV_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[SLICE_5:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_5:%.+]] = IE.Convolution([[RESHAPE_5]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice {{[^:]+}} [0, 0, 4, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_6:%.+]] = IE.Convolution([[RESHAPE_6]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x32x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_3:%.+]] = IE.Add([[CONV_5]], [[CONV_6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[ADD_0]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.Reshape([[ADD_2]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.Reshape([[ADD_3]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_7]], [[RESHAPE_8]], [[RESHAPE_9]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x812xf16>, tensor<1x32x1x812xf16>, tensor<1x32x1x812xf16> -> tensor<1x32x3x812xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 28, 29]} : tensor<1x32x3x812xf16> -> tensor<1x32x3x28x29xf16>

    // CHECK:       return [[RESHAPE_10]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution3DWithStride
func.func @ConvertNceOpsTo4DGroupConvolution3DWithStride(%arg0: tensor<1x32x5x28x28xf16>) -> tensor<1x32x3x28x29xf16> {
    %FILTERS = const.Declare tensor<32x1x3x1x1xf16> = dense<1.000000e+00> : tensor<32x1x3x1x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 1, 1], groups = 32 : i64, pads_begin = [1, 0, 0], pads_end = [1, 0, 1], strides = [2, 1, 1]} : tensor<1x32x5x28x28xf16>, tensor<32x1x3x1x1xf16> -> tensor<1x32x3x28x29xf16>

    return %RESULT : tensor<1x32x3x28x29xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.GroupConvolution([[RESHAPE_0]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.GroupConvolution([[RESHAPE_1]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.GroupConvolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 2, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_3:%.+]] = IE.GroupConvolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_4:%.+]] = IE.GroupConvolution([[RESHAPE_4]], [[CST_WEIGHTS_2]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_2:%.+]] = IE.Add([[ADD_1]], [[CONV_4]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[SLICE_5:%.+]] = IE.Slice {{[^:]+}} [0, 0, 3, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_5:%.+]] = IE.GroupConvolution([[RESHAPE_5]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[SLICE_6:%.+]] = IE.Slice {{[^:]+}} [0, 0, 4, 0, 0] [1, 32, 1, 28, 28] : tensor<1x32x5x28x28xf16> to tensor<1x32x1x28x28xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 32, 28, 28]} : tensor<1x32x1x28x28xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[CONV_6:%.+]] = IE.GroupConvolution([[RESHAPE_6]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x32x28x28xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x28x29xf16>
    // CHECK:       [[ADD_3:%.+]] = IE.Add([[CONV_5]], [[CONV_6]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x28x29xf16>, tensor<1x32x28x29xf16> -> tensor<1x32x28x29xf16>

    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[ADD_0]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.Reshape([[ADD_2]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.Reshape([[ADD_3]]) {shape_value = [1, 32, 1, 812]} : tensor<1x32x28x29xf16> -> tensor<1x32x1x812xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_7]], [[RESHAPE_8]], [[RESHAPE_9]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x812xf16>, tensor<1x32x1x812xf16>, tensor<1x32x1x812xf16> -> tensor<1x32x3x812xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 28, 29]} : tensor<1x32x3x812xf16> -> tensor<1x32x3x28x29xf16>

    // CHECK:       return [[RESHAPE_10]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateHW
func.func @ConvertNceOpsTo4DConvolution5DAggregateHW(%arg0: tensor<1x1x16x16x64xf16>) -> tensor<1x1x16x16x64xf16> {
    %FILTERS = const.Declare tensor<1x1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x1x2x1x1xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1,1,1], pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x1x16x16x64xf16>, tensor<1x1x2x1x1xf16> -> tensor<1x1x16x16x64xf16>
    return %RESULT : tensor<1x1x16x16x64xf16>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 1, 16, 1024]} : tensor<1x1x16x16x64xf16> -> tensor<1x1x16x1024xf16>
    // CHECK:       %[[CST:.*]] = const.Declare tensor<1x1x2x1xf16> = dense<1.000000e+00> : tensor<1x1x2x1x1xf16>, [#const.Reshape<[1, 1, 2, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x1x16x1024xf16>, tensor<1x1x2x1xf16> -> tensor<1x1x16x1024xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 16, 16, 64]} : tensor<1x1x16x1024xf16> -> tensor<1x1x16x16x64xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateDH
func.func @ConvertNceOpsTo4DConvolution5DAggregateDH(%arg0: tensor<1x16x16x16x16xf16>) -> tensor<1x16x16x16x16xf16> {
    %FILTERS = const.Declare tensor<16x16x1x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1x1xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1,1,1], pads_begin = [0,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x16x16x16x16xf16>, tensor<16x16x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    return %RESULT : tensor<1x16x16x16x16xf16>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 16, 256, 16]} : tensor<1x16x16x16x16xf16> -> tensor<1x16x256x16xf16>
    // CHECK:       %[[CST:.*]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1x1xf16>, [#const.Reshape<[16, 16, 1, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x256x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x256x16xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 16, 16, 16, 16]} : tensor<1x16x256x16xf16> -> tensor<1x16x16x16x16xf16>
    // CHECK:       return %[[RESULT]]
}


// -----

// Check that we would not try to flatten the tensor if the strides are not 1
// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DNonOneStrides
func.func @ConvertNceOpsTo4DConvolution5DNonOneStrides(%arg0: tensor<1x256x16x28x28xf16>) -> tensor<1x512x16x14x14xf16> {
    %FILTERS = const.Declare tensor<512x256x1x1x1xf16> = dense<1.000000e+00> : tensor<512x256x1x1x1xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [1,1,1], pads_begin = [0,0,0], pads_end = [0,0,0], strides = [1,2,2]} : tensor<1x256x16x28x28xf16>, tensor<512x256x1x1x1xf16> -> tensor<1x512x16x14x14xf16>
    return %RESULT : tensor<1x512x16x14x14xf16>

    // CHECK: %[[VAL0:.*]] = IE.Convolution
    // CHECK-NOT:       tensor<1x256x16x784xf16>
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-NOT:       strides = [1, 2]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution5DAggregateHW
func.func @ConvertNceOpsTo4DGroupConvolution5DAggregateHW(%arg0: tensor<1x2x16x16x64xf16>) -> tensor<1x2x16x16x64xf16> {
    %FILTERS = const.Declare tensor<2x1x2x1x1xf16> = dense<1.000000e+00> : tensor<2x1x2x1x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1,1,1], groups = 2 : i64, pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x2x16x16x64xf16>, tensor<2x1x2x1x1xf16> -> tensor<1x2x16x16x64xf16>
    return %RESULT : tensor<1x2x16x16x64xf16>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [1, 2, 16, 1024]} : tensor<1x2x16x16x64xf16> -> tensor<1x2x16x1024xf16>
    // CHECK:       %[[CST:.*]] = const.Declare tensor<2x1x2x1xf16> = dense<1.000000e+00> : tensor<2x1x2x1x1xf16>, [#const.Reshape<[2, 1, 2, 1]>]
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2 : i64
    // CHECK-SAME:      pads_begin = [1, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x2x16x1024xf16>, tensor<2x1x2x1xf16> -> tensor<1x2x16x1024xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 2, 16, 16, 64]} : tensor<1x2x16x1024xf16> -> tensor<1x2x16x16x64xf16>
    // CHECK:       return %[[RESULT]]

}

// -----

// CHECK-LABEL: @UnrollConvolution3Dto2DwithFQ
func.func @UnrollConvolution3Dto2DwithFQ(%arg0: tensor<1x1x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<32x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<32x1x1x1x1xf16> = dense<1.270000e+02>: tensor<32x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<32x1x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x56x56xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x3x3xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<32x1x1x1x1xf16>, tensor<32x1x1x1x1xf16> -> tensor<32x1x1x3x3xf16>
    %RESULT = IE.Convolution(%5, %6) {dilations = [1, 1, 1], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x1x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>

    return %RESULT : tensor<1x32x2x28x28xf16>

    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_4:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_5:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<-1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_6:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>

    // CHECK-DAG:   [[CST_7:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_8:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_9:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_10:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_7]], [[CST_8]], [[CST_9]], [[CST_10]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[FQ_2]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>

    // CHECK-DAG:   [[CST_11:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_12:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_13:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_14:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_3:%.*]] = IE.FakeQuantize([[RESHAPE_2]], [[CST_11]], [[CST_12]], [[CST_13]], [[CST_14]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[FQ_3]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[CONV_2]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_3]], [[RESHAPE_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>

    // CHECK:       return [[RESHAPE_5]]

}

// -----

// CHECK-LABEL: @UnrollConvolution3Dto2DwithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x2x56x56xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @UnrollConvolution3Dto2DwithDQ(%arg0: tensor<1x1x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x56x56xf16>

    %cst_2 = const.Declare tensor<32x1x1x3x3x!qElemType> = dense<1.0> : tensor<32x1x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<32x1x1x3x3x!qElemType> -> tensor<32x1x1x3x3xf16>
    
    %RESULT = IE.Convolution(%FQ, %DQ) {dilations = [1, 1, 1], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x1x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>
    return %RESULT : tensor<1x32x2x28x28xf16>

    // CHECK:  [[CST_0:%.+]] = const.Declare tensor<32x1x3x3x!qElemType> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[32, 1, 3, 3]>]
    // CHECK:  [[DQ:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<32x1x3x3x!qElemType> -> tensor<32x1x3x3xf16>
    
    // CHECK:  [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:  [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    
    // CHECK:  [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[FQ_0:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_1]], [[CST_2]], [[CST_3]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:  [[CONV_0:%.+]] = IE.Convolution([[FQ_0]], [[DQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    
    // CHECK:  [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:  [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    
    // CHECK:  [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_6:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_7:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[CST_8:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:  [[FQ_1:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_5]], [[CST_6]], [[CST_7]], [[CST_8]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:  [[CONV_1:%.+]] = IE.Convolution([[FQ_1]], [[DQ]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    
    // CHECK:  [[RESHAPE_2:%.+]] = IE.Reshape([[CONV_0]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:  [[RESHAPE_3:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:  [[CONCAT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK:  [[RESHAPE_4:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>

    // CHECK:  return [[RESHAPE_4]] : tensor<1x32x2x28x28xf16>
}

// -----

// CHECK-LABEL: @UnrollGroupConvolution3Dto2DwithFQ
func.func @UnrollGroupConvolution3Dto2DwithFQ(%arg0: tensor<1x32x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<32x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<32x1x1x1x1xf16> = dense<1.270000e+02>: tensor<32x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<32x1x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x1x3x3xf32>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]

    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x32x2x56x56xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x3x3xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<32x1x1x1x1xf16>, tensor<32x1x1x1x1xf16> -> tensor<32x1x1x3x3xf16>
    %RESULT = IE.GroupConvolution(%5, %6) {dilations = [1, 1, 1], groups = 32 : i64, pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x32x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>

    return %RESULT : tensor<1x32x2x28x28xf16>

    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK-DAG:   [[CST_WEIGHTS_2:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.000000e+00> : tensor<32x1x3x3xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_3:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_4:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_5:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<-1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_WEIGHTS_6:%.*]] = const.Declare tensor<32x1x1x1xf16> = dense<1.270000e+02> : tensor<32x1x1x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>

    // CHECK-DAG:   [[CST_7:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_8:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_9:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_10:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_7]], [[CST_8]], [[CST_9]], [[CST_10]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.GroupConvolution([[FQ_2]], [[FQ_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>

    // CHECK-DAG:   [[CST_11:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_12:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_13:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_14:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[FQ_3:%.*]] = IE.FakeQuantize([[RESHAPE_2]], [[CST_11]], [[CST_12]], [[CST_13]], [[CST_14]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.GroupConvolution([[FQ_3]], [[FQ_1]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[CONV_2]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_3]], [[RESHAPE_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>

    // CHECK:       return [[RESHAPE_5]]
}

// -----

// CHECK-LABEL: @UnrollGroupConvolution3Dto2DwithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x2x56x56xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @UnrollGroupConvolution3Dto2DwithDQ(%arg0: tensor<1x32x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x32x2x56x56xf16>

    %cst_2 = const.Declare tensor<32x1x1x3x3x!qElemType> = dense<1.0> : tensor<32x1x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<32x1x1x3x3x!qElemType> -> tensor<32x1x1x3x3xf16>
    
    %RESULT = IE.GroupConvolution(%FQ, %DQ) {dilations = [1, 1, 1], groups = 32 : i64, pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x32x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>
    return %RESULT : tensor<1x32x2x28x28xf16>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<32x1x3x3x!qElemType> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[32, 1, 3, 3]>]
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<32x1x3x3x!qElemType> -> tensor<32x1x3x3xf16>
    
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>
    
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_1]], [[CST_2]], [[CST_3]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK: [[CONV_0:%.+]] = IE.GroupConvolution([[FQ_0]], [[DQ]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>
    
    // CHECK: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_6:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_7:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_8:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_5]], [[CST_6]], [[CST_7]], [[CST_8]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK: [[CONV_1:%.+]] = IE.GroupConvolution([[FQ_1]], [[DQ]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>
    
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV_0]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[CONV_1]]) {shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>
    
    // CHECK: return [[RESHAPE_4]] : tensor<1x32x2x28x28xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateHWWithFQ
func.func @ConvertNceOpsTo4DConvolution5DAggregateHWWithFQ(%arg0: tensor<1x1x16x16x64xf16>) -> tensor<1x1x16x16x64xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02>: tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<1x1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x1x2x1x1xf16>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x16x16x64xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x1x2x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x1x1xf16>

    %RESULT = IE.Convolution(%5, %6) {dilations = [1,1,1], pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x1x16x16x64xf16>, tensor<1x1x2x1x1xf16> -> tensor<1x1x16x16x64xf16>
    return %RESULT : tensor<1x1x16x16x64xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<1x1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x1x2x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ_0:%.*]] = IE.FakeQuantize({{[^:]+}}, [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x16x16x64xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[CST_3]], [[CST]], [[CST_0]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x1x2x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x1x1xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.Reshape([[FQ_0]]) {shape_value = [1, 1, 16, 1024]} : tensor<1x1x16x16x64xf16> -> tensor<1x1x16x1024xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape([[FQ_1]]) {shape_value = [1, 1, 2, 1]} : tensor<1x1x2x1x1xf16> -> tensor<1x1x2x1xf16>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x1x16x1024xf16>, tensor<1x1x2x1xf16> -> tensor<1x1x16x1024xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 16, 16, 64]} : tensor<1x1x16x1024xf16> -> tensor<1x1x16x16x64xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateHWWithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1x16x16x64xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @ConvertNceOpsTo4DConvolution5DAggregateHWWithDQ(%arg0: tensor<1x1x16x16x64xf16>) -> tensor<1x1x16x16x64xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x16x16x64xf16>

    %cst_2 = const.Declare tensor<1x1x2x1x1x!qElemType> = dense<1.0> : tensor<1x1x2x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<1x1x2x1x1x!qElemType> -> tensor<1x1x2x1x1xf16>
    
    %RESULT = IE.Convolution(%FQ, %DQ) {dilations = [1,1,1], pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x1x16x16x64xf16>, tensor<1x1x2x1x1xf16> -> tensor<1x1x16x16x64xf16>
    return %RESULT : tensor<1x1x16x16x64xf16>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x16x16x64xf16>
    
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<1x1x2x1x1x!qElemType> = dense<1.000000e+00> : tensor<1x1x2x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[CST_2]]) {dstElemType = f16} : tensor<1x1x2x1x1x!qElemType> -> tensor<1x1x2x1x1xf16>
    
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[FQ]]) {shape_value = [1, 1, 16, 1024]} : tensor<1x1x16x16x64xf16> -> tensor<1x1x16x1024xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[DQ]]) {shape_value = [1, 1, 2, 1]} : tensor<1x1x2x1x1xf16> -> tensor<1x1x2x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE_0]], [[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x16x1024xf16>, tensor<1x1x2x1xf16> -> tensor<1x1x16x1024xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 1, 16, 16, 64]} : tensor<1x1x16x1024xf16> -> tensor<1x1x16x16x64xf16>
    
    // CHECK: return [[RESHAPE_2]] : tensor<1x1x16x16x64xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateDHWithFQ
func.func @ConvertNceOpsTo4DConvolution5DAggregateDHWithFQ(%arg0: tensor<1x16x16x16x16xf16>) -> tensor<1x16x16x16x16xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02>: tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<16x16x1x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x16x16x16xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x16x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<16x16x1x1x1xf16>

    %RESULT = IE.Convolution(%5, %6) {dilations = [1,1,1], pads_begin = [0,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x16x16x16x16xf16>, tensor<16x16x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    return %RESULT : tensor<1x16x16x16x16xf16>
    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<16x16x1x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ_0:%.*]] = IE.FakeQuantize({{[^:]+}}, [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x16x16x16xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[CST_3]], [[CST]], [[CST_0]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<16x16x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<16x16x1x1x1xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.Reshape([[FQ_0]]) {shape_value = [1, 16, 256, 16]} : tensor<1x16x16x16x16xf16> -> tensor<1x16x256x16xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape([[FQ_1]]) {shape_value = [16, 16, 1, 1]} : tensor<16x16x1x1x1xf16> -> tensor<16x16x1x1xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x256x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x256x16xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 16, 16, 16, 16]} : tensor<1x16x256x16xf16> -> tensor<1x16x16x16x16xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution5DAggregateDHWithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x16x16x16xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @ConvertNceOpsTo4DConvolution5DAggregateDHWithDQ(%arg0: tensor<1x16x16x16x16xf16>) -> tensor<1x16x16x16x16xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x16x16x16xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x16x16x16x16xf16>

    %cst_2 = const.Declare tensor<16x16x1x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<16x16x1x1x1x!qElemType> -> tensor<16x16x1x1x1xf16>
    
    %RESULT = IE.Convolution(%FQ, %DQ) {dilations = [1,1,1], pads_begin = [0,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x16x16x16x16xf16>, tensor<16x16x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    return %RESULT : tensor<1x16x16x16x16xf16>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x16x16x16xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x16x16x16x16xf16>
    
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<16x16x1x1x1x!qElemType> = dense<1.000000e+00> : tensor<16x16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[CST_2]]) {dstElemType = f16} : tensor<16x16x1x1x1x!qElemType> -> tensor<16x16x1x1x1xf16>
    
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[FQ]]) {shape_value = [1, 16, 256, 16]} : tensor<1x16x16x16x16xf16> -> tensor<1x16x256x16xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[DQ]]) {shape_value = [16, 16, 1, 1]} : tensor<16x16x1x1x1xf16> -> tensor<16x16x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE_0]], [[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x256x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x256x16xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 16, 16, 16, 16]} : tensor<1x16x256x16xf16> -> tensor<1x16x16x16x16xf16>
    
    // CHECK: return [[RESHAPE_2]] : tensor<1x16x16x16x16xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution5DAggregateHWWithFQ
func.func @ConvertNceOpsTo4DGroupConvolution5DAggregateHWWithFQ(%arg0: tensor<1x2x16x16x64xf16>) -> tensor<1x2x16x16x64xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02>: tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<2x1x2x1x1xf16> = dense<1.000000e+00> : tensor<2x1x2x1x1xf16>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x2x16x16x64xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<2x1x2x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<2x1x2x1x1xf16>

    %RESULT = IE.GroupConvolution(%5, %6) {dilations = [1,1,1], groups = 2 : i64, pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x2x16x16x64xf16>, tensor<2x1x2x1x1xf16> -> tensor<1x2x16x16x64xf16>
    return %RESULT : tensor<1x2x16x16x64xf16>
    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_2:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_3:%.*]] = const.Declare tensor<2x1x2x1x1xf16> = dense<1.000000e+00> : tensor<2x1x2x1x1xf16>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_4:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_5:%.*]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[FQ_0:%.*]] = IE.FakeQuantize({{[^:]+}}, [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x2x16x16x64xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[CST_3]], [[CST]], [[CST_0]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<2x1x2x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<2x1x2x1x1xf16>

    // CHECK:       %[[RESHAPE_0:.*]] = IE.Reshape([[FQ_0]]) {shape_value = [1, 2, 16, 1024]} : tensor<1x2x16x16x64xf16> -> tensor<1x2x16x1024xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape([[FQ_1]]) {shape_value = [2, 1, 2, 1]} : tensor<2x1x2x1x1xf16> -> tensor<2x1x2x1xf16>

    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2 : i64
    // CHECK-SAME:      pads_begin = [1, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x2x16x1024xf16>, tensor<2x1x2x1xf16> -> tensor<1x2x16x1024xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 2, 16, 16, 64]} : tensor<1x2x16x1024xf16> -> tensor<1x2x16x16x64xf16>
    // CHECK:       return %[[RESULT]]

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution5DAggregateHWWithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x2x16x16x64xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @ConvertNceOpsTo4DGroupConvolution5DAggregateHWWithDQ(%arg0: tensor<1x2x16x16x64xf16>) -> tensor<1x2x16x16x64xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x2x16x16x64xf16>

    %cst_2 = const.Declare tensor<2x1x2x1x1x!qElemType> = dense<1.0> : tensor<2x1x2x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<2x1x2x1x1x!qElemType> -> tensor<2x1x2x1x1xf16>
 
    %RESULT = IE.GroupConvolution(%FQ, %DQ) {dilations = [1,1,1], groups = 2 : i64, pads_begin = [1,0,0], pads_end = [0,0,0], strides = [1,1,1]} : tensor<1x2x16x16x64xf16>, tensor<2x1x2x1x1xf16> -> tensor<1x2x16x16x64xf16>
    return %RESULT : tensor<1x2x16x16x64xf16>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2x16x16x64xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x2x16x16x64xf16>
    
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<2x1x2x1x1x!qElemType> = dense<1.000000e+00> : tensor<2x1x2x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[CST_2]]) {dstElemType = f16} : tensor<2x1x2x1x1x!qElemType> -> tensor<2x1x2x1x1xf16>
    
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[FQ]]) {shape_value = [1, 2, 16, 1024]} : tensor<1x2x16x16x64xf16> -> tensor<1x2x16x1024xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[DQ]]) {shape_value = [2, 1, 2, 1]} : tensor<2x1x2x1x1xf16> -> tensor<2x1x2x1xf16>
    // CHECK: [[CONV:%.+]] = IE.GroupConvolution([[RESHAPE_0]], [[RESHAPE_1]]) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x16x1024xf16>, tensor<2x1x2x1xf16> -> tensor<1x2x16x1024xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[CONV]]) {shape_value = [1, 2, 16, 16, 64]} : tensor<1x2x16x1024xf16> -> tensor<1x2x16x16x64xf16>
    
    // CHECK: return [[RESHAPE_2]] : tensor<1x2x16x16x64xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3D
func.func @ConvertNceOpsTo4DTransposedConvolution3D(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x3x5x5xf16> {
    %FILTERS = const.Declare tensor<32x32x2x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0, 0],
        pads_begin = [0, 0, 0],
        pads_end = [0, 0, 0],
        strides = [1, 1, 1]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x3x5x5xf16>

    return %RESULT : tensor<1x32x3x5x5xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_0:%.+]] = IE.TransposedConvolution([[RESHAPE_0]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_1:%.+]] = IE.TransposedConvolution([[RESHAPE_1]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_2:%.+]] = IE.TransposedConvolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_3:%.+]] = IE.TransposedConvolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[ADD_12:%.+]] = IE.Add([[TCONV_1]], [[TCONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x5x5xf16>, tensor<1x32x5x5xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[TCONV_0]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[ADD_12]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[TCONV_3]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16> -> tensor<1x32x3x25xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 5, 5]} : tensor<1x32x3x25xf16> -> tensor<1x32x3x5x5xf16>

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3DWithFQ
func.func @ConvertNceOpsTo4DTransposedConvolution3DWithFQ(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x3x5x5xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_2 = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.270000e+02> : tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_3 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.270000e+02>: tensor<1x1x1x1x1xf16>, [#const.CastElemType<f16>]
    %cst_4 = const.Declare tensor<32x32x2x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_6 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %5 = IE.FakeQuantize(%arg0, %cst_6, %cst_5, %cst_6, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x4x4xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x32x2x4x4xf16>
    %6 = IE.FakeQuantize(%cst_4, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x32x2x2x2xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<32x32x2x2x2xf16>

    %RESULT = IE.TransposedConvolution(%5, %6) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0, 0],
        pads_begin = [0, 0, 0],
        pads_end = [0, 0, 0],
        strides = [1, 1, 1]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x3x5x5xf16>

    return %RESULT : tensor<1x32x3x5x5xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:       [[FQ_CST_0:%.*]] = IE.FakeQuantize([[CST_WEIGHTS_0]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x32x2x2xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:       [[FQ_CST_1:%.*]] = IE.FakeQuantize([[CST_WEIGHTS_1]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x32x2x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[FQ_0:%.*]] = IE.FakeQuantize([[RESHAPE_0]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_0:%.+]] = IE.TransposedConvolution([[FQ_0]], [[FQ_CST_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[FQ_1:%.*]] = IE.FakeQuantize([[RESHAPE_1]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_1:%.+]] = IE.TransposedConvolution([[FQ_1]], [[FQ_CST_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[FQ_2:%.*]] = IE.FakeQuantize([[RESHAPE_2]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_2:%.+]] = IE.TransposedConvolution([[FQ_2]], [[FQ_CST_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[FQ_3:%.*]] = IE.FakeQuantize([[RESHAPE_3]]
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_3:%.+]] = IE.TransposedConvolution([[FQ_3]], [[FQ_CST_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[ADD_12:%.+]] = IE.Add([[TCONV_1]], [[TCONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x5x5xf16>, tensor<1x32x5x5xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[TCONV_0]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[ADD_12]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[TCONV_3]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16> -> tensor<1x32x3x25xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 5, 5]} : tensor<1x32x3x25xf16> -> tensor<1x32x3x5x5xf16>

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3DWithDQ
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x32x2x4x4xf16>
!qElemType = !quant.uniform<u8:f16, 0.57450980392156858>
func.func @ConvertNceOpsTo4DTransposedConvolution3DWithDQ(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x3x5x5xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.CastElemType<f16>]
    %FQ = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x2x4x4xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x32x2x4x4xf16>
    
    %cst_2 = const.Declare tensor<32x32x2x2x2x!qElemType> = dense<1.0> : tensor<32x32x2x2x2xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %DQ = IE.Dequantize(%cst_2) {dstElemType = f16} : tensor<32x32x2x2x2x!qElemType> -> tensor<32x32x2x2x2xf16>
 
    %RESULT = IE.TransposedConvolution(%FQ, %DQ) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0, 0],
        pads_begin = [0, 0, 0],
        pads_end = [0, 0, 0],
        strides = [1, 1, 1]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x3x5x5xf16>

    return %RESULT : tensor<1x32x3x5x5xf16>

    // CHECK: [[CST_0:%.+]] = const.Declare tensor<32x32x2x2x!qElemType> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>, [#const.SubView<[0, 0, 0, 0, 0], [32, 32, 1, 2, 2]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[32, 32, 2, 2]>]
    // CHECK: [[DQ_0:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<32x32x2x2x!qElemType> -> tensor<32x32x2x2xf16>
    
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<32x32x2x2x!qElemType> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>, [#const.SubView<[0, 0, 0, 0, 0], [32, 32, 1, 2, 2]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reshape<[32, 32, 2, 2]>]
    // CHECK: [[DQ_1:%.+]] = IE.Dequantize([[CST_1]]) {dstElemType = f16} : tensor<32x32x2x2x!qElemType> -> tensor<32x32x2x2xf16>
    
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK: [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK: [[TCONV_0:%.+]] = IE.TransposedConvolution([[FQ_0]], [[DQ_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    
    // CHECK: [[CST_6:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_7:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_8:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_9:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[CST_6]], [[CST_7]], [[CST_8]], [[CST_9]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK: [[TCONV_1:%.+]] = IE.TransposedConvolution([[FQ_1]], [[DQ_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    
    // CHECK: [[SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    
    // CHECK: [[CST_10:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_11:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_12:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_13:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_2:%.+]] = IE.FakeQuantize([[RESHAPE_2]], [[CST_10]], [[CST_11]], [[CST_12]], [[CST_13]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK: [[TCONV_2:%.+]] = IE.TransposedConvolution([[FQ_2]], [[DQ_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    
    // CHECK: [[SLICE_3:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    
    // CHECK: [[CST_14:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_15:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_16:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.2472086> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[CST_17:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.31203485> : tensor<1x1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK: [[FQ_3:%.+]] = IE.FakeQuantize([[RESHAPE_3]], [[CST_14]], [[CST_15]], [[CST_16]], [[CST_17]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x4x4xf16>
    // CHECK: [[TCONV_3:%.+]] = IE.TransposedConvolution([[FQ_3]], [[DQ_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    
    // CHECK: [[ADD:%.+]] = IE.Add([[TCONV_1]], [[TCONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x5x5xf16>, tensor<1x32x5x5xf16> -> tensor<1x32x5x5xf16>
    // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[TCONV_0]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK: [[RESHAPE_5:%.+]] = IE.Reshape([[ADD]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK: [[RESHAPE_6:%.+]] = IE.Reshape([[TCONV_3]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16> -> tensor<1x32x3x25xf16>
    // CHECK: [[RESHAPE_7:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 3, 5, 5]} : tensor<1x32x3x25xf16> -> tensor<1x32x3x5x5xf16>
    
    // CHECK: return [[RESHAPE_7]] : tensor<1x32x3x5x5xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3DWithPad
func.func @ConvertNceOpsTo4DTransposedConvolution3DWithPad(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x1x5x5xf16> {
    %FILTERS = const.Declare tensor<32x32x2x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0, 0],
        pads_begin = [1, 0, 0],
        pads_end = [1, 0, 0],
        strides = [1, 1, 1]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x1x5x5xf16>

    return %RESULT : tensor<1x32x1x5x5xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_0:%.+]] = IE.TransposedConvolution([[RESHAPE_0]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_1:%.+]] = IE.TransposedConvolution([[RESHAPE_1]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_2:%.+]] = IE.TransposedConvolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_3:%.+]] = IE.TransposedConvolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[ADD_12:%.+]] = IE.Add([[TCONV_1]], [[TCONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x5x5xf16>, tensor<1x32x5x5xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[ADD_12]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_5]])  {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x25xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 1, 5, 5]} : tensor<1x32x1x25xf16> -> tensor<1x32x1x5x5xf16>

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3DWithStrides
func.func @ConvertNceOpsTo4DTransposedConvolution3DWithStrides(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x4x8x8xf16> {
    %FILTERS = const.Declare tensor<32x32x2x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [0, 0, 0],
        pads_begin = [0, 0, 0],
        pads_end = [0, 0, 0],
        strides = [2, 2, 2]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x4x8x8xf16>

    return %RESULT : tensor<1x32x4x8x8xf16>

    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_0:%.+]] = IE.TransposedConvolution([[RESHAPE_0]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x8x8xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_1:%.+]] = IE.TransposedConvolution([[RESHAPE_1]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x8x8xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_2:%.+]] = IE.TransposedConvolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x8x8xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_3:%.+]] = IE.TransposedConvolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x8x8xf16>

    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[TCONV_0]]) {shape_value = [1, 32, 1, 64]} : tensor<1x32x8x8xf16> -> tensor<1x32x1x64xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[TCONV_1]]) {shape_value = [1, 32, 1, 64]} : tensor<1x32x8x8xf16> -> tensor<1x32x1x64xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[TCONV_2]]) {shape_value = [1, 32, 1, 64]} : tensor<1x32x8x8xf16> -> tensor<1x32x1x64xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[TCONV_3]]) {shape_value = [1, 32, 1, 64]} : tensor<1x32x8x8xf16> -> tensor<1x32x1x64xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]], [[RESHAPE_7]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16>, tensor<1x32x1x64xf16> -> tensor<1x32x4x64xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 32, 4, 8, 8]} : tensor<1x32x4x64xf16> -> tensor<1x32x4x8x8xf16>

}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DTransposedConvolution3DWithOutputPadInvalid
func.func @ConvertNceOpsTo4DTransposedConvolution3DWithOutputPadInvalid(%arg0: tensor<1x32x2x4x4xf16>) -> tensor<1x32x4x5x5xf16> {
    %FILTERS = const.Declare tensor<32x32x2x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2x2xf16>
    %RESULT = IE.TransposedConvolution(%arg0, %FILTERS) {
        dilations = [1, 1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 0>,
        spatial_output_padding = [1, 0, 0],
        pads_begin = [0, 0, 0],
        pads_end = [0, 0, 0],
        strides = [1, 1, 1]} : tensor<1x32x2x4x4xf16>, tensor<32x32x2x2x2xf16> -> tensor<1x32x4x5x5xf16>

    return %RESULT : tensor<1x32x4x5x5xf16>


    // CHECK-DAG:   [[CST_WEIGHTS_0:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>
    // CHECK-DAG:   [[CST_WEIGHTS_1:%.*]] = const.Declare tensor<32x32x2x2xf16> = dense<1.000000e+00> : tensor<32x32x2x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_0:%.+]] = IE.TransposedConvolution([[RESHAPE_0]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice {{[^:]+}} [0, 0, 0, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_1:%.+]] = IE.TransposedConvolution([[RESHAPE_1]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_2:%.+]] = IE.TransposedConvolution([[RESHAPE_2]], [[CST_WEIGHTS_0]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice {{[^:]+}} [0, 0, 1, 0, 0] [1, 32, 1, 4, 4] : tensor<1x32x2x4x4xf16> to tensor<1x32x1x4x4xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 32, 4, 4]} : tensor<1x32x1x4x4xf16> -> tensor<1x32x4x4xf16>
    // CHECK:       [[TCONV_3:%.+]] = IE.TransposedConvolution([[RESHAPE_3]], [[CST_WEIGHTS_1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [1, 1]} : tensor<1x32x4x4xf16>, tensor<32x32x2x2xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[ADD_12:%.+]] = IE.Add([[TCONV_1]], [[TCONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x32x5x5xf16>, tensor<1x32x5x5xf16> -> tensor<1x32x5x5xf16>

    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[TCONV_0]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[ADD_12]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[TCONV_3]]) {shape_value = [1, 32, 1, 25]} : tensor<1x32x5x5xf16> -> tensor<1x32x1x25xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16>, tensor<1x32x1x25xf16> -> tensor<1x32x3x25xf16>
    // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]} : tensor<1x32x3x25xf16> -> tensor<1x32x4x25xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[EXPAND]]) {shape_value = [1, 32, 4, 5, 5]} : tensor<1x32x4x25xf16> -> tensor<1x32x4x5x5xf16>
}
