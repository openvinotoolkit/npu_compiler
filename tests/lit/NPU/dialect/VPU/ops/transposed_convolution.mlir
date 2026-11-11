//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
!dynOutputType = tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

// CHECK: func.func @TransposedConvolutionDynamicInputConstFilter([[INPUT_DATA:%.+]]: tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}> {
func.func @TransposedConvolutionDynamicInputConstFilter(%input: !dynInputType) -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}> {
    %weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : !dynInputType, tensor<16x32x2x2xf16, {order = #NHWC}> -> !dynOutputType
    return %output : !dynOutputType

    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          spatial_output_padding = [0, 0],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynInputType = tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
!dynOutputType = tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

// CHECK: func.func @TransposedConvolutionDynamicInput([[INPUT_DATA:%.+]]: tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>, [[FILTER:%.+]]: tensor<16x32x2x2xf16, {order = #NHWC}>) -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}> {
func.func @TransposedConvolutionDynamicInput(%input: !dynInputType, %weights: tensor<16x32x2x2xf16, {order = #NHWC}>) -> !dynOutputType {
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : !dynInputType, tensor<16x32x2x2xf16, {order = #NHWC}> -> !dynOutputType
    return %output : !dynOutputType

    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          spatial_output_padding = [0, 0],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x32x23x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 23, 30]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!dynFilterType = tensor<16x32x2x?xf16, {bounds = #const.OpaqueI64Elements<[16, 32, 2, 2]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
!dynOutputType = tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

// CHECK: func.func @TransposedConvolutionDynamicFilter([[INPUT_DATA:%.+]]: tensor<1x32x23x30xf16, {order = #NHWC}>, [[FILTER:%.+]]: tensor<16x32x2x?xf16, {bounds = #const.OpaqueI64Elements<[16, 32, 2, 2]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}> {
func.func @TransposedConvolutionDynamicFilter(%input: tensor<1x32x23x30xf16, {order = #NHWC}>, %weights: !dynFilterType) -> !dynOutputType {
    %output = VPU.TransposedConvolution(%input, %weights) {
            dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
        } : tensor<1x32x23x30xf16, {order = #NHWC}>, !dynFilterType -> !dynOutputType
    return %output : !dynOutputType

    // CHECK:       [[TRANSPOSED_CONV:%.+]] = VPU.TransposedConvolution([[INPUT_DATA]], [[FILTER]]) {
    // CHECK-SAME:          dilations = [1, 1],
    // CHECK-SAME:          operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:          pads_begin = [0, 0],
    // CHECK-SAME:          pads_end = [0, 0],
    // CHECK-SAME:          spatial_output_padding = [0, 0],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x32x23x30xf16, {order = #NHWC}>, tensor<16x32x2x?xf16, {bounds = #const.OpaqueI64Elements<[16, 32, 2, 2]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:       return [[TRANSPOSED_CONV]] : tensor<1x16x24x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 24, 31]> : tensor<4xsi64>, order = #NHWC}>
}
