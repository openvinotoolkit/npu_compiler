//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-conv3d-to-conv2d --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010
// COM: F8 is only supported on NPU50+, no need to run these tests on all platforms.

// CHECK-LABEL: @UnrollConvolution3Dto2DwithDynDeqF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x2x56x56xf16>
func.func @UnrollConvolution3Dto2DwithDynDeqF8E4M3FN(%input: tensor<1x1x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %weights = const.Declare tensor<32x1x1x3x3xf8E4M3FN> = dense<1.000000e+00> : tensor<32x1x1x3x3xf8E4M3FN>
    %scale = const.Declare tensor<32x1x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1x1xf16>

    %in_low = const.Declare tensor<1x1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1x1xf16>
    %in_high = const.Declare tensor<1x1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %in_low, %in_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E4M3FN
    } : tensor<1x1x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x56x56xf16>

    %1 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<32x1x1x3x3xf8E4M3FN>, tensor<32x1x1x1x1xf16> -> tensor<32x1x1x3x3xf16>

    %2 = IE.Convolution(%0, %1) {dilations = [1, 1, 1], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x1x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>

    return %2 : tensor<1x32x2x28x28xf16>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x1x3x3xf8E4M3FN> = dense<1.000000e+00> : tensor<32x1x1x3x3xf8E4M3FN>, [#const.Reshape<[32, 1, 3, 3]>]
    // CHECK-DAG:    [[SCALE:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1x1xf16>, [#const.Reshape<[32, 1, 1, 1]>]
    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]

    // CHECK:    [[DDQ_0:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16} : tensor<32x1x3x3xf8E4M3FN>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>

    // CHECK:    [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:    [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E4M3FN} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:    [[CONV_0:%.+]] = IE.Convolution([[FQ_1]], [[DDQ_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 1, 1, 56, 56] : tensor<1x1x2x56x56xf16> to tensor<1x1x1x56x56xf16>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 1, 56, 56]} : tensor<1x1x1x56x56xf16> -> tensor<1x1x56x56xf16>
    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E4M3FN} : tensor<1x1x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x56x56xf16>
    // CHECK:    [[CONV_1:%.+]] = IE.Convolution([[FQ_2]], [[DDQ_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x1x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:    [[RESHAPE_2:%.+]] = IE.AffineReshape([[CONV_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:    [[RESHAPE_3:%.+]] = IE.AffineReshape([[CONV_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>

    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_3]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK:    [[RESHAPE_4:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>

    // CHECK:    return [[RESHAPE_4]] : tensor<1x32x2x28x28xf16>
}

// -----

// CHECK-LABEL: @UnrollGroupConvolution3Dto2DwithDynDeqF8E5M2
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x2x56x56xf16>
func.func @UnrollGroupConvolution3Dto2DwithDynDeqF8E5M2(%input: tensor<1x32x2x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %weights = const.Declare tensor<32x1x1x3x3xf8E5M2> = dense<1.000000e+00> : tensor<32x1x1x3x3xf8E5M2>
    %scale = const.Declare tensor<32x1x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1x1xf16>

    %in_low = const.Declare tensor<1x1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1x1xf16>
    %in_high = const.Declare tensor<1x1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %in_low, %in_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E5M2
    } : tensor<1x32x2x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x32x2x56x56xf16>

    %1 = IE.DynamicDequantize(%weights, %scale) {dstElemType = f16} : tensor<32x1x1x3x3xf8E5M2>, tensor<32x1x1x1x1xf16> -> tensor<32x1x1x3x3xf16>

    %2 = IE.GroupConvolution(%0, %1) {dilations = [1, 1, 1], groups = 32 : i64, pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]} : tensor<1x32x2x56x56xf16>, tensor<32x1x1x3x3xf16> -> tensor<1x32x2x28x28xf16>

    return %2 : tensor<1x32x2x28x28xf16>

    // CHECK-DAG:    [[WEIGHTS:%.+]] = const.Declare tensor<32x1x3x3xf8E5M2> = dense<1.000000e+00> : tensor<32x1x1x3x3xf8E5M2>, [#const.Reshape<[32, 1, 3, 3]>]
    // CHECK-DAG:    [[SCALE:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x1x1x1xf16>, [#const.Reshape<[32, 1, 1, 1]>]
    // CHECK-DAG:    [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:    [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]

    // CHECK:    [[DDQ_0:%.+]] = IE.DynamicDequantize([[WEIGHTS]], [[SCALE]]) {dstElemType = f16} : tensor<32x1x3x3xf8E5M2>, tensor<32x1x1x1xf16> -> tensor<32x1x3x3xf16>

    // CHECK:    [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK:    [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>
    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[RESHAPE_0]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E5M2} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK:    [[CONV_0:%.+]] = IE.GroupConvolution([[FQ_1]], [[DDQ_0]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:    [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 1, 0, 0] [1, 32, 1, 56, 56] : tensor<1x32x2x56x56xf16> to tensor<1x32x1x56x56xf16>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 32, 56, 56]} : tensor<1x32x1x56x56xf16> -> tensor<1x32x56x56xf16>
    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[RESHAPE_1]], [[IN_LOW]], [[IN_HIGH]], [[IN_LOW]], [[IN_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E5M2} : tensor<1x32x56x56xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x56x56xf16>
    // CHECK:    [[CONV_1:%.+]] = IE.GroupConvolution([[FQ_2]], [[DDQ_0]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x56x56xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x28x28xf16>

    // CHECK:    [[RESHAPE_2:%.+]] = IE.AffineReshape([[CONV_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>
    // CHECK:    [[RESHAPE_3:%.+]] = IE.AffineReshape([[CONV_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 32, 1, 784]} : tensor<1x32x28x28xf16> -> tensor<1x32x1x784xf16>

    // CHECK:    [[CONCAT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_3]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0]]} : tensor<1x32x1x784xf16>, tensor<1x32x1x784xf16> -> tensor<1x32x2x784xf16>
    // CHECK:    [[RESHAPE_4:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 32, 2, 28, 28]} : tensor<1x32x2x784xf16> -> tensor<1x32x2x28x28xf16>

    // CHECK:    return [[RESHAPE_4]] : tensor<1x32x2x28x28xf16>
}

// -----

// CHECK-LABEL: @UnrollConvolution3Dto2DwithPerDepthDynDeqParams
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x3x56x56xf16>
func.func @UnrollConvolution3Dto2DwithPerDepthDynDeqParams(%input: tensor<1x1x3x56x56xf16>) -> tensor<1x32x2x28x28xf16> {
    %weights = const.Declare tensor<32x1x2x3x3xsi8> = dense<1> : tensor<32x1x2x3x3xsi8>
    %scale = const.Declare tensor<32x1x2x1x1xf16> = dense<1.000000e+00> : tensor<32x1x2x1x1xf16>
    %zp = const.Declare tensor<32x1x2x1x1xsi8> = dense<2> : tensor<32x1x2x1x1xsi8>

    %in_low = const.Declare tensor<1x1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1x1xf16>
    %in_high = const.Declare tensor<1x1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %in_low, %in_high, %in_low, %in_high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = f8E4M3FN
    } : tensor<1x1x3x56x56xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x3x56x56xf16>

    %1 = IE.DynamicDequantize(%weights, %scale, %zp) {dstElemType = f16, vpux.weights_import_dyn_dequant}
        : tensor<32x1x2x3x3xsi8>, tensor<32x1x2x1x1xf16>, tensor<32x1x2x1x1xsi8> -> tensor<32x1x2x3x3xf16>

    %2 = IE.Convolution(%0, %1) {dilations = [1, 1, 1], pads_begin = [0, 1, 1], pads_end = [0, 1, 1], strides = [1, 2, 2]}
        : tensor<1x1x3x56x56xf16>, tensor<32x1x2x3x3xf16> -> tensor<1x32x2x28x28xf16>

    return %2 : tensor<1x32x2x28x28xf16>

    // CHECK-DAG:    [[WEIGHTS_0:%.+]] = const.Declare tensor<32x1x3x3xsi8> = dense<1> : tensor<32x1x2x3x3xsi8>, [#const.SubView<[0, 0, 0, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]
    // CHECK-DAG:    [[SCALE_0:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x2x1x1xf16>, [#const.SubView<[0, 0, 0, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]
    // CHECK-DAG:    [[ZP_0:%.+]] = const.Declare tensor<32x1x1x1xsi8> = dense<2> : tensor<32x1x2x1x1xsi8>, [#const.SubView<[0, 0, 0, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]

    // CHECK-DAG:    [[WEIGHTS_1:%.+]] = const.Declare tensor<32x1x3x3xsi8> = dense<1> : tensor<32x1x2x3x3xsi8>, [#const.SubView<[0, 0, 1, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]
    // CHECK-DAG:    [[SCALE_1:%.+]] = const.Declare tensor<32x1x1x1xf16> = dense<1.000000e+00> : tensor<32x1x2x1x1xf16>, [#const.SubView<[0, 0, 1, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]
    // CHECK-DAG:    [[ZP_1:%.+]] = const.Declare tensor<32x1x1x1xsi8> = dense<2> : tensor<32x1x2x1x1xsi8>, [#const.SubView<[0, 0, 1, 0, 0], {{.*}}>, #const.Reshape<{{.*}}>]

    // CHECK:        [[DDQ_0:%.+]] = IE.DynamicDequantize([[WEIGHTS_0]], [[SCALE_0]], [[ZP_0]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<32x1x3x3xsi8>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xsi8> -> tensor<32x1x3x3xf16>
    // CHECK:        [[DDQ_1:%.+]] = IE.DynamicDequantize([[WEIGHTS_1]], [[SCALE_1]], [[ZP_1]]) {dstElemType = f16, vpux.weights_import_dyn_dequant} : tensor<32x1x3x3xsi8>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xsi8> -> tensor<32x1x3x3xf16>

    // CHECK:        [[CONV_0:%.+]] = IE.Convolution({{.*}}, [[DDQ_0]])
    // CHECK:        [[CONV_1:%.+]] = IE.Convolution({{.*}}, [[DDQ_1]])
}
