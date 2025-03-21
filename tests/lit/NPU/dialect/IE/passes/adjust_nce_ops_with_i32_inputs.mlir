//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-nce-ops-with-i32-inputs --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvSI32toFP16
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x128x64xsi32, {order = #NHWC}>
func.func @ConvSI32toFP16(%arg0: tensor<1x1x128x64xsi32, {order = #NHWC}>) -> tensor<1x2x64x64xsi32, {order = #NHWC}> {
    %cst = const.Declare tensor<2x1x2x1xsi32, {order = #NHWC}> = dense<1> : tensor<2x1x2x1xsi32, {order = #NHWC}>
    %1 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [2, 1]} :
        tensor<1x1x128x64xsi32, {order = #NHWC}>, tensor<2x1x2x1xsi32, {order = #NHWC}> -> tensor<1x2x64x64xsi32, {order = #NHWC}>

    return %1 : tensor<1x2x64x64xsi32, {order = #NHWC}>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<2x1x2x1xf16, {order = #NHWC}> = dense<1> : tensor<2x1x2x1xsi32, {order = #NHWC}>, [#const.CastElemType<f16>]
    // CHECK:       [[INPUT_CVT:%.+]] = IE.Convert({{[^:]+}}) {dstElemType = f16} : tensor<1x1x128x64xsi32, {order = #NHWC}> -> tensor<1x1x128x64xf16, {order = #NHWC}>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT_CVT]], [[FILTERS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [2, 1]} : tensor<1x1x128x64xf16, {order = #NHWC}>, tensor<2x1x2x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x2x64x64xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_CVT:%.+]] = IE.Convert([[CONV]]) {dstElemType = si32} : tensor<1x2x64x64xf16, {order = #NHWC}> -> tensor<1x2x64x64xsi32, {order = #NHWC}>

    // CHECK:       return [[OUTPUT_CVT]] : tensor<1x2x64x64xsi32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @GroupConvSI32toFP16
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x300x300xsi32>
func.func @GroupConvSI32toFP16(%arg0: tensor<1x16x300x300xsi32>) -> tensor<1x16x300x300xsi32> {
    %filters = const.Declare tensor<16x1x1x3x3xsi32> = dense<1> : tensor<16x1x1x3x3xsi32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x16x300x300xsi32>, tensor<16x1x1x3x3xsi32> -> tensor<1x16x300x300xsi32>

    return %0 : tensor<1x16x300x300xsi32>

    // CHECK-DAG:   [[FILTERS:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1> : tensor<16x1x1x3x3xsi32>, [#const.Reshape<[16, 1, 3, 3]>, #const.CastElemType<f16>]
    // CHECK:       [[INPUT_CVT:%.+]] = IE.Convert({{[^:]+}}) {dstElemType = f16} : tensor<1x16x300x300xsi32> -> tensor<1x16x300x300xf16>
    // CHECK:       [[GROUP_CONV:%.+]] = IE.GroupConvolution([[INPUT_CVT]], [[FILTERS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      strides = [1, 1]} : tensor<1x16x300x300xf16>, tensor<16x1x3x3xf16>
    // CHECK-SAME:      -> tensor<1x16x300x300xf16>
    // CHECK:       [[OUTPUT_CVT:%.+]] = IE.Convert([[GROUP_CONV]]) {dstElemType = si32} : tensor<1x16x300x300xf16> -> tensor<1x16x300x300xsi32>
    // CHECK:       return [[OUTPUT_CVT]] : tensor<1x16x300x300xsi32>
}

// -----

// CHECK-LABEL: @MatMul_SI32toFP16
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x20x560xsi32>
func.func @MatMul_SI32toFP16(%arg0: tensor<1x20x560xsi32>) -> tensor<1x20x1536xsi32> {
    %cst = const.Declare tensor<1536x560xsi32> = dense<1> : tensor<1536x560xsi32>
    %0 = IE.MatMul(%arg0, %cst) {transpose_b} : tensor<1x20x560xsi32>, tensor<1536x560xsi32> -> tensor<1x20x1536xsi32>
    return %0 : tensor<1x20x1536xsi32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1536x560xf16> = dense<1> : tensor<1536x560xsi32>, [#const.CastElemType<f16>]
    // CHECK: [[INPUT_CVT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x20x560xsi32> -> tensor<1x20x560xf16>
    // CHECK: [[MAT_MUL:%.+]] = IE.MatMul([[INPUT_CVT]], [[CST]]) {transpose_b} : tensor<1x20x560xf16>, tensor<1536x560xf16> -> tensor<1x20x1536xf16>
    // CHECK: [[OUT_CVT:%.+]] = IE.Convert([[MAT_MUL]]) {dstElemType = si32} : tensor<1x20x1536xf16> -> tensor<1x20x1536xsi32>
    // CHECK: return [[OUT_CVT]] : tensor<1x20x1536xsi32>
}

// -----

// CHECK-LABEL: @FC_SI32toFP16
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<2x3xsi32>, [[INPUT1:%.+]]: tensor<2x3xsi32>
func.func @FC_SI32toFP16(%arg0: tensor<2x3xsi32>, %arg1: tensor<2x3xsi32>) -> tensor<2x2xsi32> {
    %0 = IE.FullyConnected(%arg0, %arg1) : tensor<2x3xsi32>, tensor<2x3xsi32> -> tensor<2x2xsi32>
    return %0 : tensor<2x2xsi32>

    // CHECK:    [[INPUT_CVT0:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f16} : tensor<2x3xsi32> -> tensor<2x3xf16>
    // CHECK:    [[INPUT_CVT1:%.+]] = IE.Convert([[INPUT1]]) {dstElemType = f16} : tensor<2x3xsi32> -> tensor<2x3xf16>
    // CHECK:    [[FC:%.+]] = IE.FullyConnected([[INPUT_CVT0]], [[INPUT_CVT1]])
    // CHECK-SAME:      : tensor<2x3xf16>, tensor<2x3xf16> -> tensor<2x2xf16>
    // CHECK:    [[OUT_CVT:%.+]] = IE.Convert([[FC]]) {dstElemType = si32} : tensor<2x2xf16> -> tensor<2x2xsi32>
    // CHECK:    return [[OUT_CVT]] : tensor<2x2xsi32>
}
