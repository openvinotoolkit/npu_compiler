//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --multi-cluster-strategy-assignment %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: NCEConvolutionOp has unsupported batch size and cannot be assigned SOB strategy
// Batch size 6 is not supported for SOB strategy on NPU40XX+ with 2 tiles enabled.
func.func @ConvAssignedSOBWith2Tiles(%arg0: tensor<6x1024x14x14xf16, {order = #NHWC}>) -> tensor<6x256x14x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x1x1x4xsi32> = dense<10> : tensor<256x1x1x4xsi32>
    %cst_0 = const.Declare tensor<256x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [256, 1024, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,  strides = [1, 1]} : tensor<6x1024x14x14xf16, {order = #NHWC}>, tensor<256x1024x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<6x256x14x14xf16, {order = #NHWC}>
    return %0 : tensor<6x256x14x14xf16, {order = #NHWC}>
}

// -----

config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: NCEMaxPoolOp has unsupported batch size and cannot be assigned SOB strategy
// Batch size 6 is not supported for SOB strategy on NPU40XX+ with 2 tiles enabled.
func.func @MaxPoolAssignedSOBWith2Tiles(%input: tensor<6x32x112x112xf16, {order = #NHWC}>) -> tensor<6x32x112x112xf16, {order = #NHWC}> {
    %maxpool = VPU.NCE.MaxPool(%input) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x112x112xf16, {order = #NHWC}>
    return %maxpool : tensor<6x32x112x112xf16, {order = #NHWC}>
}

// -----

config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: NCEAveragePoolOp has unsupported batch size and cannot be assigned SOB strategy
// Batch size 6 is not supported for SOB strategy on NPU40XX+ with 2 tiles enabled.
func.func @AveragePoolAssignedSOBWith2Tiles(%input: tensor<6x32x112x112xf16, {order = #NHWC}>) -> tensor<6x32x112x112xf16, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%input) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        kernel_size = [1, 1]
    } -> tensor<6x32x112x112xf16, {order = #NHWC}>
    return %avgpool : tensor<6x32x112x112xf16, {order = #NHWC}>
}
