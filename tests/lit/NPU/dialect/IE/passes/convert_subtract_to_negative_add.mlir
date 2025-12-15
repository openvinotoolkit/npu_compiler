//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-subtract-to-add --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SubtractIntInput
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x128x128xsi32>
func.func @SubtractIntInput(%INPUT: tensor<1x1x128x128xsi32>) -> tensor<1x1x128x128xsi32> {
    %cst_0 = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.CastElemType<si32>]
    %3 = IE.Subtract(%cst_0, %INPUT) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x128x128xsi32> -> tensor<1x1x128x128xsi32>

    return %3 : tensor<1x1x128x128xsi32>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi64>, [#const.CastElemType<si32>]
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[CST]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xsi32>, tensor<1x1x128x128xsi32> -> tensor<1x1x128x128xsi32>
    // CHECK:       return [[SUBTRACT]]
}

// -----

// CHECK-LABEL: @SubtractFP32InputAndNeedConvert2GroupConv
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf32>
func.func @SubtractFP32InputAndNeedConvert2GroupConv(%arg0: tensor<1x16x32x1xf32>) -> tensor<1x16x32x1xf32> {
    %cst = const.Declare tensor<1x16x1x1xf32> = dense<2.0> : tensor<1x16x1x1xf32>
    %input_2 = IE.ReLU(%cst) : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf32>
    %0 = IE.Subtract(%arg0, %input_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>

    return %0 : tensor<1x16x32x1xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf32> = dense<2.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[CST]]) : tensor<1x16x1x1xf32> -> tensor<1x16x1x1xf32>
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[INPUT]], [[RELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x32x1xf32>
}

// -----

// CHECK-LABEL: @SubtractWithConstAtSecondInputDiffShapes
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @SubtractWithConstAtSecondInputDiffShapes(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<2.0> : tensor<1x16x1x1xf16>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x1xf16>

    return %0 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK-SAME:      #const.Rescale<-1.000000e+00 : f64>
    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithSplatConstAtFirstInputConvertToBias
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SubtractWithSplatConstAtFirstInputConvertToBias(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>
    %0 = IE.Subtract(%cst, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @SubtractWithNonSplatConstAtFirstInputConvertToBias
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SubtractWithNonSplatConstAtFirstInputConvertToBias(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<[[[[1.0]], [[2.0]]]]> : tensor<1x2x1x1xf16>, [#const.Broadcast<1 : i64, 16 : i64>]
    %0 = IE.Subtract(%cst, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @SubtractWithConstAtFirstInputNotConvertToBias
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SubtractWithConstAtFirstInputNotConvertToBias(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<1x1x1x32xf16> = dense<2.0> : tensor<1x1x1x32xf16>
    %0 = IE.Subtract(%cst, %arg0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x32xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<2.000000e+00> : tensor<1x1x1x32xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_0]]) {
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x32x32xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithSplatConstFQAtFirstInputNotConvertToBias
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SubtractWithSplatConstFQAtFirstInputNotConvertToBias(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %2 = IE.FakeQuantize(%arg0, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    %0 = IE.Subtract(%1, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK-DAG:       [[FQ_VAL_MAX_2_NEGATED:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FILTER_FQ_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK-DAG:       [[ADD_CONST_INPUT:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>

    // CHECK:       [[ADD_INPUT_1:%.+]] = IE.FakeQuantize([[ADD_CONST_INPUT]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_1]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_1]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[CONV_FQ_INPUT:%.+]] = IE.FakeQuantize([[INPUT]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_2]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[CONV_FQ_FILTER:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_FQ_LOW]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FILTER_FQ_LOW]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV_FQ_INPUT]], [[CONV_FQ_FILTER]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD_INPUT_2:%.+]] = IE.FakeQuantize([[GROUP_CONV]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithSplatConstDiffFQRangAtFirstInputNotConvertToBias
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x32xf16>
func.func @SubtractWithSplatConstDiffFQRangAtFirstInputNotConvertToBias(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-01> : tensor<1x1x1x1xf16>
    %cst_3 = const.Declare tensor<1x1x1x1xf16> = dense<2.574580e-01> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %2 = IE.FakeQuantize(%arg0, %cst_0, %cst_2, %cst_0, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    %0 = IE.Subtract(%1, %2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x1x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>

    return %0 : tensor<1x16x32x32xf16>

    // CHECK-DAG:       [[FQ_VAL_MAX_3_NEGATED:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-2.575680e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2_NEGATED:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.575200e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FILTER_FQ_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK-DAG:       [[ADD_CONST_INPUT:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.575200e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.575680e-01> : tensor<1x1x1x1xf16>

    // CHECK:       [[ADD_INPUT_1:%.+]] = IE.FakeQuantize([[ADD_CONST_INPUT]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_1]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[CONV_FQ_INPUT:%.+]] = IE.FakeQuantize([[INPUT]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_2]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_3]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[CONV_FQ_FILTER:%.+]] = IE.FakeQuantize([[FILTER]], [[FILTER_FQ_LOW]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FILTER_FQ_LOW]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV_FQ_INPUT]], [[CONV_FQ_FILTER]])
    // CHECK-SAME:      dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD_INPUT_2:%.+]] = IE.FakeQuantize([[GROUP_CONV]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]], [[FQ_VAL_MAX_3_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_FQ_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x16x32x32xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithConstInputsSameShapes
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @SubtractWithConstInputsSameShapes(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x16x32x1xf16> = dense<2.0> : tensor<1x16x32x1xf16>
    %0 = IE.Subtract(%arg0, %cst)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf16>, tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>

    return %0 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x16x32x1xf16> = dense<2.000000e+00> : tensor<1x16x32x1xf16>
    // CHECK-SAME:      #const.Rescale<-1.000000e+00 : f64>
    // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsBroadcast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @NotConvertSubtractWithInputNeedsBroadcast(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x16x1x1xf16> = dense<2.0> : tensor<1x16x1x1xf16>
    %input_2 = IE.ReLU(%cst) : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %0 = IE.Subtract(%arg0, %input_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x1xf16>

    return %0 : tensor<1x16x32x1xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[RELU:%.+]] = IE.ReLU([[CST]]) : tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK:       [[SUBTRACT:%.+]] = IE.Subtract([[INPUT]], [[RELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x1xf16>
}

// -----

// CHECK-LABEL: @SubtractWithActivationInputsSameShapes
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @SubtractWithActivationInputsSameShapes(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %input_1 = IE.ReLU(%arg0) : tensor <1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    %cst = const.Declare tensor<1x16x32x1xf16> = dense<2.0> : tensor<1x16x32x1xf16>
    %input_2 = IE.ReLU(%cst) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    %0 = IE.Subtract(%input_1, %input_2)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x16x32x1xf16>, tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>

    return %0 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[FILTER:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<-1.000000e+00> : tensor<16x1x1x1xf16>
    // CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1x16x32x1xf16> = dense<2.000000e+00> : tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[RELU_1:%.+]] = IE.ReLU([[INPUT]]) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[RELU_2:%.+]] = IE.ReLU([[CST2]]) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[RELU_2]], [[FILTER]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x16x32x1xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[RELU_1]], [[GROUP_CONV]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithFQConstInputsSameShape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x1x64xf16>
func.func @SubtractWithFQConstInputsSameShape(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %2 = IE.FakeQuantize(%cst, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %3 : tensor<1x1x1x64xf16>

    // CHECK-DAG:       [[CONST:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK-DAG:       [[VAL_LOW_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_LOW_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_1:%.+]] = IE.FakeQuantize([[INPUT]], [[VAL_LOW_2]], [[VAL_HIGH_1]], [[VAL_LOW_2]], [[VAL_HIGH_1]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[ADD_INPUT_2:%.+]] = IE.FakeQuantize([[CONST]], [[VAL_LOW_1]], [[VAL_LOW_2]], [[VAL_LOW_1]], [[VAL_LOW_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[ADD:%.+]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:          tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @SubtractWithFQConstInputsDiffShape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @SubtractWithFQConstInputsDiffShape(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    %2 = IE.FakeQuantize(%cst, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>

    return %3 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CONST:%.+]] = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
    // CHECK-DAG:       [[VAL_LOW_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[VAL_HIGH_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_1:%.+]] = IE.FakeQuantize([[INPUT]], [[VAL_HIGH_2]], [[VAL_HIGH_1]], [[VAL_HIGH_2]], [[VAL_HIGH_1]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[ADD_INPUT_2:%.+]] = IE.FakeQuantize([[CONST]], [[VAL_LOW_2]], [[VAL_HIGH_2]], [[VAL_LOW_2]], [[VAL_HIGH_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:          tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    // CHECK-DAG:       [[ADD:%.+]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:          tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithFQActInputNeedsBroadcast
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x32x1xf16>
func.func @NotConvertSubtractWithFQActInputNeedsBroadcast(%arg0: tensor<1x16x32x1xf16>) -> tensor<1x16x32x1xf16> {
    %cst = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    %act_1 = IE.ReLU(%arg0) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    %act_2 = IE.ReLU(%cst) : tensor<1x1x32x1xf16> -> tensor<1x1x32x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%act_1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x16x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x32x1xf16>
    %2 = IE.FakeQuantize(%act_2, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x32x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x32x1xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>

    return %3 : tensor<1x16x32x1xf16>

    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[CST_3:%.+]] = const.Declare tensor<1x1x32x1xf16> = dense<5.000000e+00> : tensor<1x1x32x1xf16>
    // CHECK-DAG:       [[RELU_0:%.+]] = IE.ReLU([[INPUT]]) : tensor<1x16x32x1xf16> -> tensor<1x16x32x1xf16>
    // CHECK-DAG:       [[RELU_1:%.+]] = IE.ReLU([[CST_3]]) : tensor<1x1x32x1xf16> -> tensor<1x1x32x1xf16>
    // CHECK:           [[FQ_0:%.+]] = IE.FakeQuantize([[RELU_0]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]])
    // CHECK:           [[FQ_1:%.+]] = IE.FakeQuantize([[RELU_1]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]])
    // CHECK:           [[SUB:%.+]] = IE.Subtract([[FQ_0]], [[FQ_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x1xf16>, tensor<1x1x32x1xf16> -> tensor<1x16x32x1xf16>
}

// -----

// CHECK-LABEL: @SubtractWithFQActInputsSameShape
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x1x64xf16>
func.func @SubtractWithFQActInputsSameShape(%arg0: tensor<1x1x1x64xf16>) -> tensor<1x1x1x64xf16> {
    %cst = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    %act_1 = IE.ReLU(%arg0) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    %act_2 = IE.ReLU(%cst) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%act_1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %2 = IE.FakeQuantize(%act_2, %cst_0, %cst_2, %cst_0, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    %3 = IE.Subtract(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>

    return %3 : tensor<1x1x1x64xf16>

    // CHECK-DAG:       [[FILTER_AND_FILTER_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2_NEGATED:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MIN_AND_FILTER_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.574580e-02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[FQ_VAL_MAX_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    // CHECK-DAG:       [[RELU_2_INPUT:%.+]] = const.Declare tensor<1x1x1x64xf16> = dense<5.000000e+00> : tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[RELU_1:%.+]] = IE.ReLU([[INPUT]]) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK-DAG:       [[RELU_2:%.+]] = IE.ReLU([[RELU_2_INPUT]]) : tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD_INPUT_1:%.+]] = IE.FakeQuantize([[RELU_1]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FQ_VAL_MAX_1]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FQ_VAL_MAX_1]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[CONV_FQ_INPUT:%.+]] = IE.FakeQuantize([[RELU_2]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FQ_VAL_MAX_2]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FQ_VAL_MAX_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[CONV_FQ_FILTER:%.+]] = IE.FakeQuantize([[FILTER_AND_FILTER_LOW]], [[FILTER_AND_FILTER_LOW]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FILTER_AND_FILTER_LOW]], [[FQ_VAL_MIN_AND_FILTER_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[CONV_FQ_INPUT]], [[CONV_FQ_FILTER]])
    // CHECK-SAME:      dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD_INPUT_2:%.+]] = IE.FakeQuantize([[GROUP_CONV]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_HIGH]], [[FQ_VAL_MAX_2_NEGATED]], [[FQ_VAL_MIN_AND_FILTER_HIGH]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       [[ADD:%.+]] = IE.Add([[ADD_INPUT_1]], [[ADD_INPUT_2]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:      tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:       return [[ADD]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsBroadcast
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1x8192xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<1x1x1x1xf16>
func.func @NotConvertSubtractWithInputNeedsBroadcast(%arg0: tensor<1x1x1x8192xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x8192xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x8192xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x8192xf16>
    return %0 : tensor<1x1x1x8192xf16>

    // CHECK:  [[SUB:%.+]] = IE.Subtract([[INPUT_1]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x8192xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x8192xf16>
    // CHECK:  return [[SUB]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractWithInputNeedsExpansion
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x10x1x57xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<1x10x1x57xf16>
func.func @NotConvertSubtractWithInputNeedsExpansion(%arg0: tensor<1x10x1x57xf16>, %arg1: tensor<1x10x1x57xf16>) -> tensor<1x10x1x57xf16> {
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1x57xf16>, tensor<1x10x1x57xf16> -> tensor<1x10x1x57xf16>
    return %0 : tensor<1x10x1x57xf16>

    // CHECK:  [[SUB:%.+]] = IE.Subtract([[INPUT_1]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x1x57xf16>, tensor<1x10x1x57xf16> -> tensor<1x10x1x57xf16>
    // CHECK:  return [[SUB]]
}

// -----

// CHECK-LABEL: @ConvertSubtractIfChannelExpansionNotRequired
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x1x16xf16>
func.func @ConvertSubtractIfChannelExpansionNotRequired(%arg0: tensor<1x1x1x16xf16>) -> tensor<1x1x1x16xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Subtract(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x1x16xf16>
    return %0 : tensor<1x1x1x16xf16>

    // CHECK:  [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<1 : i64, 1 : i64>]
    // CHECK:  [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:  [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_1]], [[CST_0]]) {dilations = [1, 1], groups = 1 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x1x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x16xf16>
    // CHECK:  return [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @NotConvertSubtractIfChannelExpansionRequired
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x1x4xf16>
func.func @NotConvertSubtractIfChannelExpansionRequired(%arg0: tensor<1x1x1x4xf16>) -> tensor<1x1x1x4xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = IE.Subtract(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x4xf16> -> tensor<1x1x1x4xf16>
    return %0 : tensor<1x1x1x4xf16>

    // CHECK:  [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:  [[SUB:%.+]] = IE.Subtract([[CST]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x4xf16> -> tensor<1x1x1x4xf16>
    // CHECK:  return [[SUB]]
}

// -----

// CHECK-LABEL: @ConvertSubtractWithConstantMultiplyInput
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x1x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1x1xf16>
func.func @ConvertSubtractWithConstantMultiplyInput(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %mul = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %sub = IE.Subtract(%arg1, %mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    return %sub : tensor<1x1x1x1xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>,
  // CHECK-SAME:    [#const.Rescale<-1.000000e+00 : f64>]
  // CHECK: [[MUL:%.+]] = IE.Multiply([[INPUT_0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:    tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK: [[ADD:%.+]] = IE.Add([[INPUT_1]], [[MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:    tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK: return [[ADD]] : tensor<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @ConvertSubtractWithFQConstantMultiplyInput
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x1x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1x1xf16>
func.func @ConvertSubtractWithFQConstantMultiplyInput(%arg0: tensor<1x1x1x1xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %fa_val_min = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %fa_val_max = const.Declare tensor<1x1x1x1xf16> = dense<3.862300e-01> : tensor<1x1x1x1xf16>
    %1 = IE.FakeQuantize(%cst, %fa_val_min, %fa_val_max, %fa_val_min, %fa_val_max) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %mul = IE.Multiply(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    %sub = IE.Subtract(%arg1, %mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    return %sub : tensor<1x1x1x1xf16>

  // CHECK: [[FQ_VAL_MIN:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-3.862300e-01> : tensor<1x1x1x1xf16>
  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Rescale<-1.000000e+00 : f64>]
  // CHECK: [[FQ_VAL_MAX:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>

  // CHECK:       [[MUL_CST_INPUT:%.+]] = IE.FakeQuantize([[CST]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX]], [[FQ_VAL_MIN]], [[FQ_VAL_MAX]])
  // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
  // CHECK-SAME:      tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

  // CHECK: [[MUL:%.+]] = IE.Multiply([[MUL_CST_INPUT]], [[INPUT_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:                tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK: [[ADD:%.+]] = IE.Add([[INPUT_1]], [[MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:                tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>
  // CHECK: return [[ADD]] : tensor<1x1x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @DynamicSubtract
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicSubtract(%arg0: tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense_resource<vpux_ow_1> : tensor<1x1x1x1xf32> isSplat
    %0 = IE.Subtract(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense_resource<vpux_ow_1> : tensor<1x1x1x1xf32> isSplat
  // CHECK: [[SUB:%.+]] = IE.Subtract([[CST]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:    tensor<1x1x1x1xf32>, tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>
  // CHECK: return [[SUB]] : tensor<1x1x1x?xf32, {bounds = #const.OpaqueI64Elements<[1, 1, 1, 64]> : tensor<4xsi64>, order = #NCHW}>

}

{-#
  dialect_resources: {
    builtin: {
      vpux_ow_1: "0x1000000000004040"
    }
  }
#-}
