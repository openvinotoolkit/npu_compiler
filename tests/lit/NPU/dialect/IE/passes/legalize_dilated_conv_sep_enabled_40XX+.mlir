//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-dilated-conv="enable-sep-dilated-group-conv=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// CHECK-LABEL: @DontLegalizeDilatedGroupConvolution
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x3x30x30xf16>
func.func @DontLegalizeDilatedGroupConvolution(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x30x30xf16> {
    %filter = const.Declare tensor<3x1x3x3xf16> = dense<1.0> : tensor<3x1x3x3xf16>
    %0 = IE.GroupConvolution(%arg0, %filter)
        {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } :
        tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %0 : tensor<1x3x30x30xf16>

    // CHECK:       [[FILTERS:%.+]] = const.Declare tensor<3x1x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.GroupConvolution([[ARG0]], [[FILTERS]])  {dilations = [2, 2], groups = 3 : i64, pads_begin = [2, 2],
    // CHECK-SAME   pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>

    // CHECK: return [[CONV]]
}



// -----

// CHECK-LABEL: @ConvertXDilatedGroupConvolutionToGroupConvolution
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x512x1x32xf16>
func.func @ConvertXDilatedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x1x32xf16>) -> tensor<1x512x1x48xf16> {
    %FILTERS = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 8], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1x48xf16>
    return %RESULT : tensor<1x512x1x48xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[CST]] [0, 0, 0, 1] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[CST]] [0, 0, 0, 2] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.GroupConvolution([[SLICE_3]], [[SLICE_0]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 0], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_1:%.+]] = IE.GroupConvolution([[SLICE_4]], [[SLICE_1]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 8], pads_end = [0, 8], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 512, 1, 32] : tensor<1x512x1x32xf16> to tensor<1x512x1x32xf16>
    // CHECK:       [[CONV_2:%.+]] = IE.GroupConvolution([[SLICE_5]], [[SLICE_2]]) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 16], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CONV_2]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       return [[ADD_1]] : tensor<1x512x1x48xf16>
}

// -----

// CHECK-LABEL: @DontLegalizeDilatedGroupConvolutionWeightsAsInputs
// CHECK-SAME:  ([[INPUT:%.+]]: tensor<1x3x30x30xf16>, [[WEIGHTS:%.+]]: tensor<3x1x3x3xf16>)
func.func @DontLegalizeDilatedGroupConvolutionWeightsAsInputs(%input: tensor<1x3x30x30xf16>, %weights: tensor<3x1x3x3xf16>) -> tensor<1x3x30x30xf16> {
    %conv = IE.GroupConvolution(%input, %weights) {
            dilations = [2, 2],
            groups = 3,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1]
        } : tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>
    return %conv : tensor<1x3x30x30xf16>

    // CHECK:       [[GROUPCONV:%.+]] = IE.GroupConvolution([[INPUT]], [[WEIGHTS]]) {dilations = [2, 2], groups = 3 : i64, pads_begin = [2, 2],
    // CHECK-SAME:  pads_end = [2, 2], strides = [1, 1]} : tensor<1x3x30x30xf16>, tensor<3x1x3x3xf16> -> tensor<1x3x30x30xf16>

    // CHECK:       return [[GROUPCONV]]
}
