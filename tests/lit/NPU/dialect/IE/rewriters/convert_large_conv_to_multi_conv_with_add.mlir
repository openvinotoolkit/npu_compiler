//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --run-adjust-for-vpu-rewriters="rewriter=convert-large-conv-to-multi-conv-with-add" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @ConvertLargeConvToMultiConvWithAddTile2
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1280x64x64xf16>
func.func @ConvertLargeConvToMultiConvWithAddTile2(%arg0: tensor<1x1280x64x64xf16>) -> tensor<1x640x64x64xf16> {
    %filter = const.Declare tensor<640x1280x3x3xf16> = dense<0.100000e+00> : tensor<640x1280x3x3xf16>
    %bias = const.Declare tensor<1x640x1x1xf16> = dense<0.200000e+00> : tensor<1x640x1x1xf16>
    %conv = IE.Convolution(%arg0, %filter, %bias) {
                dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
            } : tensor<1x1280x64x64xf16>, tensor<640x1280x3x3xf16>, tensor<1x640x1x1xf16> -> tensor<1x640x64x64xf16>

    return %conv : tensor<1x640x64x64xf16>

    // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<640x640x3x3xf16> = dense<9.997550e-02> : tensor<640x1280x3x3xf16>, [#const.SubView<[0, 640, 0, 0], [640, 640, 3, 3]>]
    // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<640x640x3x3xf16> = dense<9.997550e-02> : tensor<640x1280x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [640, 640, 3, 3]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x640x1x1xf16> = dense<1.999510e-01> : tensor<1x640x1x1xf16>

    // CHECK:       [[IN_SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 640, 64, 64] : tensor<1x1280x64x64xf16> to tensor<1x640x64x64xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[IN_SLICE_0]], [[FILTER1]]) {
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x640x64x64xf16>, tensor<640x640x3x3xf16> -> tensor<1x640x64x64xf16>

    // CHECK:       [[IN_SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 640, 0, 0] [1, 640, 64, 64] : tensor<1x1280x64x64xf16> to tensor<1x640x64x64xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[IN_SLICE_1]], [[FILTER0]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x640x64x64xf16>, tensor<640x640x3x3xf16>, tensor<1x640x1x1xf16> -> tensor<1x640x64x64xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x640x64x64xf16>, tensor<1x640x64x64xf16> -> tensor<1x640x64x64xf16>
    // CHECK:       return [[ADD]] : tensor<1x640x64x64xf16>
}

// -----

// CHECK-LABEL: @ConvertLargeConvToMultiConvWithAddTile3
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1920x64x64xf16>
func.func @ConvertLargeConvToMultiConvWithAddTile3(%arg0: tensor<1x1920x64x64xf16>) -> tensor<1x640x64x64xf16> {
    %filter = const.Declare tensor<640x1920x3x3xf16> = dense<0.100000e+00> : tensor<640x1920x3x3xf16>
    %bias = const.Declare tensor<1x640x1x1xf16> = dense<0.200000e+00> : tensor<1x640x1x1xf16>
    %conv = IE.Convolution(%arg0, %filter, %bias) {
                dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
            } : tensor<1x1920x64x64xf16>, tensor<640x1920x3x3xf16>, tensor<1x640x1x1xf16> -> tensor<1x640x64x64xf16>

    return %conv : tensor<1x640x64x64xf16>

    // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<640x640x3x3xf16> = dense<9.997550e-02> : tensor<640x1920x3x3xf16>, [#const.SubView<[0, 1280, 0, 0], [640, 640, 3, 3]>]
    // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<640x640x3x3xf16> = dense<9.997550e-02> : tensor<640x1920x3x3xf16>, [#const.SubView<[0, 640, 0, 0], [640, 640, 3, 3]>]
    // CHECK-DAG:   [[FILTER2:%.+]] = const.Declare tensor<640x640x3x3xf16> = dense<9.997550e-02> : tensor<640x1920x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [640, 640, 3, 3]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x640x1x1xf16> = dense<1.999510e-01> : tensor<1x640x1x1xf16>

    // CHECK:       [[IN_SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 640, 64, 64] : tensor<1x1920x64x64xf16> to tensor<1x640x64x64xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[IN_SLICE_0]], [[FILTER2]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x640x64x64xf16>, tensor<640x640x3x3xf16> -> tensor<1x640x64x64xf16>

    // CHECK:       [[IN_SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 640, 0, 0] [1, 640, 64, 64] : tensor<1x1920x64x64xf16> to tensor<1x640x64x64xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[IN_SLICE_1]], [[FILTER1]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x640x64x64xf16>, tensor<640x640x3x3xf16> -> tensor<1x640x64x64xf16>

    // CHECK:       [[IN_SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 1280, 0, 0] [1, 640, 64, 64] : tensor<1x1920x64x64xf16> to tensor<1x640x64x64xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.Convolution([[IN_SLICE_2]], [[FILTER0]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x640x64x64xf16>, tensor<640x640x3x3xf16>, tensor<1x640x1x1xf16> -> tensor<1x640x64x64xf16>

    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x640x64x64xf16>, tensor<1x640x64x64xf16> -> tensor<1x640x64x64xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x640x64x64xf16>, tensor<1x640x64x64xf16> -> tensor<1x640x64x64xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x640x64x64xf16>
}

// -----

// CHECK-LABEL: @ConvertLargeFqConvToMultiConvWithAdd
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2560x64x64xf16>
func.func @ConvertLargeFqConvToMultiConvWithAdd(%arg0: tensor<1x2560x64x64xf16>) -> tensor<1x1280x64x64xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-20.00000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<20.00000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-30.00000e+00> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<30.00000e+00> : tensor<1x1x1x1xf16>
    %in_fq = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x2560x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x2560x64x64xf16>

    %filter = const.Declare tensor<1280x2560x3x3xf16> = dense<0.100000e+00> : tensor<1280x2560x3x3xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    %cst_4 = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    %cst_5 = const.Declare tensor<1280x1x1x1xf16> = dense<-40.00000e+00> : tensor<1280x1x1x1xf16>
    %cst_6 = const.Declare tensor<1280x1x1x1xf16> = dense<40.00000e+00> : tensor<1280x1x1x1xf16>
    %filter_fq = IE.FakeQuantize(%filter, %cst_3, %cst_4, %cst_5, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1280x2560x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1280x1x1x1xf16>, tensor<1280x1x1x1xf16> -> tensor<1280x2560x3x3xf16>

    %bias = const.Declare tensor<1x1280x1x1xf16> = dense<0.200000e+00> : tensor<1x1280x1x1xf16>
    %conv = IE.Convolution(%in_fq, %filter_fq, %bias) {
                dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]
            } : tensor<1x2560x64x64xf16>, tensor<1280x2560x3x3xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x64x64xf16>

    %out_fq = IE.FakeQuantize(%conv, %cst_3, %cst_4, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1280x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1280x64x64xf16>

    return %out_fq : tensor<1x1280x64x64xf16>

    // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<1280x1280x3x3xf16> = dense<9.997550e-02> : tensor<1280x2560x3x3xf16>, [#const.SubView<[0, 1280, 0, 0], [1280, 1280, 3, 3]>]
    // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<1280x1280x3x3xf16> = dense<9.997550e-02> : tensor<1280x2560x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [1280, 1280, 3, 3]>]
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x1280x1x1xf16> = dense<1.999510e-01> : tensor<1x1280x1x1xf16>

    // CHECK-DAG:   [[FILTER_OUT_HIGH:%.+]] = const.Declare tensor<1280x1x1x1xf16> = dense<4.000000e+01> : tensor<1280x1x1x1xf16>
    // CHECK-DAG:   [[FILTER_OUT_LOW:%.+]] = const.Declare tensor<1280x1x1x1xf16> = dense<-4.000000e+01> : tensor<1280x1x1x1xf16>
    // CHECK-DAG:   [[FILTER_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[FILTER_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.000000e+01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-3.000000e+01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+01> : tensor<1x1x1x1xf16>

    // CHECK:       [[IN_SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1280, 64, 64] : tensor<1x2560x64x64xf16> to tensor<1x1280x64x64xf16>
    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize([[IN_SLICE_0]], [[INPUT_IN_LOW]], [[INPUT_IN_HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1280x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1280x64x64xf16>
    // CHECK:       [[FQ_FILTER_0:%.+]] = IE.FakeQuantize([[FILTER1]], [[FILTER_IN_LOW]], [[FILTER_IN_HIGH]], [[FILTER_OUT_LOW]], [[FILTER_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1280x1280x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1280x1x1x1xf16>, tensor<1280x1x1x1xf16> -> tensor<1280x1280x3x3xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[FQ_0]], [[FQ_FILTER_0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1280x64x64xf16>, tensor<1280x1280x3x3xf16> -> tensor<1x1280x64x64xf16>

    // CHECK:       [[IN_SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 1280, 0, 0] [1, 1280, 64, 64] : tensor<1x2560x64x64xf16> to tensor<1x1280x64x64xf16>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize([[IN_SLICE_1]], [[INPUT_IN_LOW]], [[INPUT_IN_HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1280x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1280x64x64xf16>
    // CHECK:       [[FQ_FILTER_1:%.+]] = IE.FakeQuantize([[FILTER0]], [[FILTER_IN_LOW]], [[FILTER_IN_HIGH]], [[FILTER_OUT_LOW]], [[FILTER_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1280x1280x3x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1280x1x1x1xf16>, tensor<1280x1x1x1xf16> -> tensor<1280x1280x3x3xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[FQ_1]], [[FQ_FILTER_1]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x1280x64x64xf16>, tensor<1280x1280x3x3xf16>, tensor<1x1280x1x1xf16> -> tensor<1x1280x64x64xf16>

    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x1280x64x64xf16>, tensor<1x1280x64x64xf16> -> tensor<1x1280x64x64xf16>
    // CHECK:       [[FQ_RESULT:%.+]] = IE.FakeQuantize([[ADD]], [[FILTER_IN_LOW]], [[FILTER_IN_HIGH]], [[INPUT_OUT_LOW]], [[INPUT_OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1280x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1280x64x64xf16>

    // CHECK:       return [[FQ_RESULT]] : tensor<1x1280x64x64xf16>
}
