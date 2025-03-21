//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-network-input-convert %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>
!qElemType1 = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK-LABEL: @FloatInConv2d
func.func @FloatInConv2d(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1x!qElemType1> = dense<2.000000e+00> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>,
        #const.CastElemType<!qElemType1>
    ]

    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x16x1x1x!qElemType1> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    // CHECK:   [[DEQUANT:%.*]] = IE.Dequantize([[CST]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.Convolution(%arg0, [[DEQUANT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>
!qElemType1 = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK-LABEL: @FloatInGroupConv
func.func @FloatInGroupConv(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x1x1x1x!qElemType1> = dense<2.000000e+00> : tensor<16x1x1x1xf16>, [
        #const.CastElemType<ui8>,
        #const.CastElemType<!qElemType1>
    ]

    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.GroupConvolution(%0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType>, tensor<16x1x1x1x!qElemType1> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<16x1x1x1x!qElemType1> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    // CHECK:   [[DEQUANT:%.*]] = IE.Dequantize([[CST]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x1x1x1x!qElemType1> -> tensor<16x1x1x1xf16>

    // CHECK:   [[VAL0:%.*]] = IE.GroupConvolution(%arg0, [[DEQUANT]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK-LABEL: @FloatInAdd
func.func @FloatInAdd(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.Add(%0, %0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.Add(%arg0, %arg0) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK-LABEL: @FloatInAvgPool
func.func @FloatInAvgPool(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.AvgPool(%0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
// CHECK-LABEL: @SkipFloatInMaxPool
func.func @SkipFloatInMaxPool(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Quantize(%arg0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %1 = IE.MaxPool(%0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[QUANT:%.*]] = IE.Quantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.*]] = IE.MaxPool([[QUANT]]) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}
