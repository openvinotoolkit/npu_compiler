//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-outstanding-dequant %s | FileCheck %s
// REQUIRES: platform-NPU6010

!qElemType = !quant.uniform<u16:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @FuseU16WithNonReluPostOp
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x16x4x4x!qElemType>) -> tensor<1x16x4x4xf16>
func.func @FuseU16WithNonReluPostOp(%arg0: tensor<1x16x4x4x!qElemType>) -> tensor<1x16x4x4xf16> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1> : tensor<16x16x1x1xui8>
    %conv = IE.Convolution(%arg0, %weights) {
        clamp = {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64},
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x4x4x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x4x4x!qElemType>
    %dequant = IE.Dequantize(%conv) {dstElemType = f16} :
            tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4xf16>
    return %dequant : tensor<1x16x4x4xf16>

    // CHECK: [[WEIGHTS:%.+]] = const.Declare
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS]]) {clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64}
    // CHECK-NOT: IE.Dequantize
    // CHECK-NOT: IE.Clamp
    // CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u16:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @FuseDequantThroughShapeAndLayoutCasts
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x16x4x4x!qElemType>) -> tensor<1x16x4x4xf16, {order = #NHWC}>
func.func @FuseDequantThroughShapeAndLayoutCasts(%arg0: tensor<1x16x4x4x!qElemType>)
        -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1> : tensor<16x16x1x1xui8>
    %conv = IE.Convolution(%arg0, %weights) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x16x4x4x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x4x4x!qElemType>
    %shape = IE.ShapeCast {shape = [1, 16, 4, 4]} inputs(%conv : tensor<1x16x4x4x!qElemType>) ->
            tensor<1x16x4x4x!qElemType>
    %layout = IE.LayoutCast(%shape) {dst_order = #NHWC} :
            tensor<1x16x4x4x!qElemType> -> tensor<1x16x4x4x!qElemType, {order = #NHWC}>
    %dequant = IE.Dequantize(%layout) {dstElemType = f16} :
            tensor<1x16x4x4x!qElemType, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>
    return %dequant : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK: [[WEIGHTS:%.+]] = const.Declare
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG0]], [[WEIGHTS]]) {
    // CHECK: [[SHAPE:%.+]] = IE.ShapeCast
    // CHECK: [[LAYOUT:%.+]] = IE.LayoutCast([[SHAPE]]) {dst_order = #NHWC}
    // CHECK-NOT: IE.Dequantize
    // CHECK: return [[LAYOUT]]
}
