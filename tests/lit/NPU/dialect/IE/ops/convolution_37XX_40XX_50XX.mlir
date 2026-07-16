//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseTransposedConvAndBias
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x64x64xf16>)
func.func @FuseTransposedConvAndBias(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x129x129xf16> {
    %filters = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    %0 = IE.TransposedConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            spatial_output_padding = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]
        } :
        tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16> -> tensor<1x16x129x129xf16>

    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x129x129xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x129x129xf16>

    return %1 : tensor<1x16x129x129xf16>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[VAL0:%.+]] = IE.TransposedConvolution([[ARG_0]], [[FILTERS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 1>,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      spatial_output_padding = [1, 1],
    // CHECK-SAME:      strides = [2, 2]} : tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16>, tensor<1x16x1x1xf16>
    // CHECK-SAME:      -> tensor<1x16x129x129xf16>

    // CHECK:       return [[VAL0]] : tensor<1x16x129x129xf16>
}
