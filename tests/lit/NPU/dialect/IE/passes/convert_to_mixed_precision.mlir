//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

func.func @MixedPrecisionConv(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL2]]
}

// -----


!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @AvoidMixedPrecisionConvPerAxes(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[VAL4:%.+]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionGroupConv(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @MixedPrecisionGroupConvPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType1>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @MixedPrecisionAdd(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1x!qElemType>, tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

func.func @AvoidMixedPrecisionAddDifferentOperandType(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL2:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @AvoidMixedPrecisionAddPerAxes(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL2:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @AvoidMixedPrecisionAvgPoolPerAxisWithPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x3x3xf16>)
func.func @AvoidMixedPrecisionAvgPoolPerAxisWithPostOp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %avgPool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %quantize = IE.Quantize(%avgPool) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %quantize : tensor<1x16x3x3x!qElemType>

    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[ARG0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>, rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[AVGPOOL]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    // CHECK: return [[QUANTIZE]] : tensor<1x16x3x3x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @AvoidMixedPrecisionMaxPool
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x3x3xf16>)
func.func @AvoidMixedPrecisionMaxPool(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %maxPool = IE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %quantize = IE.Quantize(%maxPool) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %quantize : tensor<1x16x3x3x!qElemType>

    // CHECK: [[MAXPOOL:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[MAXPOOL]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    // CHECK: return [[QUANTIZE]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @AvoidMixedPrecisionAddPerAxisWithPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>, [[ARG1:%.+]]: tensor<1x16x1x1xf16>)
func.func @AvoidMixedPrecisionAddPerAxisWithPostOp(%arg0: tensor<1x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1x!qElemType> {
    %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %quantize = IE.Quantize(%add) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

    return %quantize : tensor<1x16x1x1x!qElemType>

    // CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ADD]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    // CHECK: return [[QUANTIZE]] : tensor<1x16x1x1x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @AvoidMixedPrecisionConvPerAxisWithPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x3x3xf16>)
func.func @AvoidMixedPrecisionConvPerAxisWithPostOp(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>
    %convolution = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>
    %quantize = IE.Quantize(%convolution) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %quantize : tensor<1x16x3x3x!qElemType>

    // CHECK: [[CONST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK: [[CONVOLUTION:%.+]] = IE.Convolution([[ARG0]], [[CONST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>, strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[CONVOLUTION]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    // CHECK: return [[QUANTIZE]] : tensor<1x16x3x3x!qElemType>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionConvForOutputShape
func.func @MixedPrecisionConvForOutputShape(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

  return %4 : tensor<1x16x3x3xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x3x3xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionGroupConvForOutputShape
func.func @MixedPrecisionGroupConvForOutputShape(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<16x1x1x1x!qElemType> = dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x1x1x!qElemType> -> tensor<16x1x1x1xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAvgPoolForOutputShape
func.func @MixedPrecisionAvgPoolForOutputShape(%arg0: tensor<1x64x88x88x!qElemType>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x88x88xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88xf16> -> tensor<1x64x11x11xf16>

    return %1 : tensor<1x64x11x11xf16>

    // CHECK-NOT: IE.Dequantize
    // CHECK:       %[[AVGPOOL:.+]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionTransposedConvForOutputShape
func.func @MixedPrecisionTransposedConvForOutputShape(%arg0: tensor<1x20x23x30xf16>) -> tensor<1x20x46x60xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x20x23x30xf16> -> tensor<1x20x23x30x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x20x23x30x!qElemType> -> tensor<1x20x23x30xf16>
    %weights = const.Declare tensor<20x20x2x2x!qElemType> = dense<1.0> : tensor<20x20x2x2xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<20x20x2x2x!qElemType> -> tensor<20x20x2x2xf16>
    %4 = IE.TransposedConvolution(%2, %3) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x20x23x30xf16>, tensor<20x20x2x2xf16> -> tensor<1x20x46x60xf16>
    return %4 : tensor<1x20x46x60xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<20x20x2x2x!qElemType>
    //CHECK-SAME:   = dense<1.000000e+00> : tensor<20x20x2x2xf16>,
    //CHECK-SAME:   [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x20x23x30xf16> -> tensor<1x20x23x30x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.TransposedConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x20x23x30x!qElemType>, tensor<20x20x2x2x!qElemType> -> tensor<1x20x46x60xf16>

    //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAddForSameScales
func.func @MixedPrecisionAddForSameScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, -1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAddForSameNegativeScale
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionAddForSameNegativeScale(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[ADD:%.+]] = IE.Add([[QUANT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3x!qElemType>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[ADD]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, 0.500000e+00>

// CHECK-LABEL: @MixedPrecisionAddForDifferentScales
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionAddForDifferentScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} :
    // CHECK-SAME:  tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>

    // CHECK: [[ADD:%.+]] = IE.Add([[QUANT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x16x3x3x!qElemType1>, tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: return [[ADD]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16, -0.500000e+00>

// CHECK-LABEL: @MixedPrecisionAddForDifferentNegativeScales
// CHECK-SAME: [[ARG0:%.+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionAddForDifferentNegativeScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    // CHECK: [[QUANT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    // CHECK: [[DQUANT_LHS:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    // CHECK: [[DQUANT_RHS:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[ADD:%.+]] = IE.Add([[DQUANT_LHS]], [[DQUANT_RHS]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:  tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[ADD]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @DoNotConvertConvWithPReLU
func.func @DoNotConvertConvWithPReLU(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %1 = IE.Quantize(%arg0) {
      dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    %2 = IE.Dequantize(%1) {
      dstElemType = f16
    } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %WEIGHTS = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [
        #const.CastElemType<ui8>, #const.CastElemType<!qElemType>
    ]

    %3 = IE.Dequantize(%WEIGHTS) {
        dstElemType = f16
    } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    %4 = IE.Convolution(%2, %3) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    return %4 : tensor<1x16x3x3xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x1x1xf16>, [
    // CHECK-SAME:      #const.CastElemType<ui8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType>
    // CHECK-SAME:  ]

    // CHECK: [[QUANT:%.+]] = IE.Quantize(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK: [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    // CHECK: [[DEQUANT_WEIGHTS:%.+]] = IE.Dequantize([[WEIGHTS]]) {
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:  } : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANT]], [[DEQUANT_WEIGHTS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    // CHECK: return [[CONV]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @Conv2dWithQuantize
func.func @Conv2dWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x1x1xf16>

    // CHECK:   [[VAL0:%.+]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @DoNotConv2dWithQuantize
func.func @DoNotConv2dWithQuantize(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x1x1x!qElemType> {
    %cst = const.Declare tensor<16x16x16x16xf16> = dense<2.000000e+00> : tensor<16x16x16x16xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x16xf16>, tensor<16x16x16x16xf16> -> tensor<1x16x1x1xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

    return %1 : tensor<1x16x1x1x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x16x16xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x16x16xf16>

    // CHECK:   [[VAL0:%.+]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x16x16xf16>, tensor<16x16x16x16xf16> -> tensor<1x16x1x1xf16>

    // CHECK:   [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

    // CHECK:   return [[VAL1]] : tensor<1x16x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @GroupConvWithQuantize
func.func @GroupConvWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1x1xf16>

    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x1x1x1xf16>

    // CHECK:   [[VAL0:%.+]] = IE.GroupConvolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 16 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @AvgPoolWithQuantize
func.func @AvgPoolWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.+]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @DoNotAvgPoolWithQuantize
func.func @DoNotAvgPoolWithQuantize(%arg0: tensor<1x1x64x128xf16>) -> tensor<1x1x1x128x!qElemType> {
    %0 = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [64, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x1x64x128xf16> -> tensor<1x1x1x128xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x1x1x128xf16> -> tensor<1x1x1x128x!qElemType>

    return %1 : tensor<1x1x1x128x!qElemType>

    // CHECK:   [[VAL0:%.+]] = IE.AvgPool(%arg0) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [64, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x1x64x128xf16> -> tensor<1x1x1x128xf16>
    //
    // CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      } : tensor<1x1x1x128xf16> -> tensor<1x1x1x128x!qElemType>
    //
    // CHECK:   return [[VAL1]] : tensor<1x1x1x128x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039005478223164878>

func.func @DoNotFuseQuantParamsIntoAvgPoolWithExcludePadsAttr(%arg0: tensor<1x3x135x240xf16> ) -> tensor<1x3x68x120x!qElemType> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType>
    return %1 : tensor<1x3x68x120x!qElemType>

    //CHECK: [[AVGPOOL:%.+]] = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
    // CHECK: [[RESULT:%.+]] = IE.Quantize([[AVGPOOL]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      } : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType>
    // CHECK: return [[RESULT]] : tensor<1x3x68x120x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @AddWithQuantize
func.func @AddWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.+]] = IE.Add(%arg0, %arg0) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionGroupConvForOutputShapeWithQuantWeightsBias
func.func @MixedPrecisionGroupConvForOutputShapeWithQuantWeightsBias(%arg0: tensor<1x3x320x480xf16>) -> tensor<1x3x320x480xf16> {
    %cst = const.Declare tensor<1x3x1x1x!qElemType> = dense<1.270000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %cst_0 = const.Declare tensor<3x1x1x1x!qElemType1> = dense<2.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x320x480xf16> -> tensor<1x3x320x480x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x3x320x480x!qElemType1> -> tensor<1x3x320x480xf16>
    %2 = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<3x1x1x1x!qElemType1> -> tensor<3x1x1x1xf16>
    %3 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>
    %4 = IE.GroupConvolution(%1, %2, %3) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x480xf16>, tensor<3x1x1x1xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x320x480xf16>
    return %4 : tensor<1x3x320x480xf16>

    //CHECK: [[CST0:%.+]] = const.Declare tensor<1x3x1x1x!qElemType> =
    //CHECK-SAME:     dense<1.270000e+02> : tensor<1x3x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    //CHECK: [[CST:%.+]] = const.Declare tensor<3x1x1x1x!qElemType1> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<3x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>]
    //CHECK: [[VAL0:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x3x320x480xf16> -> tensor<1x3x320x480x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[CST0]]) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>
    //CHECK: [[VAL2:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]], [[VAL1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x480x!qElemType1>, tensor<3x1x1x1x!qElemType1>, tensor<1x3x1x1xf16> -> tensor<1x3x320x480xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x320x480xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @DoNotAddWithQuantize
func.func @DoNotAddWithQuantize(%arg0: tensor<1x8x8x32xf16>) -> tensor<1x8x8x32x!qElemType> {
    %cst0 = const.Declare tensor<1x1x1x32xf16> = dense<1.000000e+00> : tensor<1x1x1x32xf16>
    %0 = IE.Add(%arg0, %cst0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x8x8x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x8x8x32xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x8x8x32xf16> -> tensor<1x8x8x32x!qElemType>

    return %1 : tensor<1x8x8x32x!qElemType>

    // CHECK:   [[CST0:%.+]] = const.Declare tensor<1x1x1x32xf16> = dense<1.000000e+00> : tensor<1x1x1x32xf16>
    // CHECK:   [[VAL0:%.+]] = IE.Add(%arg0, [[CST0]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x8x8x32xf16>, tensor<1x1x1x32xf16> -> tensor<1x8x8x32xf16>
    //
    // CHECK:   [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x8x8x32xf16> -> tensor<1x8x8x32x!qElemType>

    // CHECK:   return [[VAL1]] : tensor<1x8x8x32x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.01:128,0.02:128,0.03:128}>

// CHECK-LABEL: @DoNotConv2dLeakyReluWithQuantize
func.func @DoNotConv2dLeakyReluWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x3x3x3x!qElemType> {
    %cst = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> : tensor<3x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>

    return %1 : tensor<1x3x3x3x!qElemType>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<3x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<3x16x1x1xf16>

    // CHECK:   [[VAL0:%.+]] = IE.Convolution(%arg0, [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.LeakyRelu<negative_slope = 2.500000e-01 : f64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<3x16x1x1xf16> -> tensor<1x3x3x3xf16>

    // CHECK:  [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x3x3x3xf16> -> tensor<1x3x3x3x!qElemType>
    // CHECK:   return [[VAL1]] : tensor<1x3x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1368405211205576E-7>

func.func @AvoidMixedPrecisionForInvalidApproximation(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256x!qElemType> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 1 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 1.000000e+00 : f64>,
        strides = [1, 1]
    } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    return %1 : tensor<1x1x256x256x!qElemType>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    //CHECK-SAME:               dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    //CHECK: [[VAL0:%.+]] = IE.GroupConvolution(%arg0, [[CST]]) {
    //CHECK-SAME:               dilations = [1, 1],
    //CHECK-SAME:               groups = 1 : i64,
    //CHECK-SAME:               pads_begin = [0, 0],
    //CHECK-SAME:               pads_end = [0, 0],
    //CHECK-SAME:               post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 1.000000e+00 : f64>,
    //CHECK-SAME:               strides = [1, 1]
    //CHECK-SAME:           } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    //CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    //CHECK: return [[VAL1]] : tensor<1x1x256x256x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

func.func @MixedPrecisionGroupMatmul(%arg0: tensor<1x8x1x64xf16>) -> tensor<1x8x1x128xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x1x64x!qElemType> -> tensor<1x8x1x64xf16>
  %weights = const.Declare tensor<1x8x128x64x!qElemType> = dense<1.0> : tensor<1x8x128x64xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x128x64x!qElemType> -> tensor<1x8x128x64xf16>
  %4 = IE.MatMul(%2, %3) {transpose_b} : tensor<1x8x1x64xf16>, tensor<1x8x128x64xf16> -> tensor<1x8x1x128xf16>

  return %4 : tensor<1x8x1x128xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<1x8x128x64x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<1x8x128x64xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

  //CHECK: [[VAL1:%.+]] = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[VAL0]]) {transpose_b} : tensor<1x8x1x64x!qElemType>, tensor<1x8x128x64x!qElemType> -> tensor<1x8x1x128xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @DoNotFuseMixedPrecisionSubtractPerAxisOut
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>, [[ARG1:%.+]]: tensor<1x16x1x1xf16>)
func.func @DoNotFuseMixedPrecisionSubtractPerAxisOut(%arg0: tensor<1x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1x!qElemType> {
    %sub = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %quantize = IE.Quantize(%sub) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

    return %quantize : tensor<1x16x1x1x!qElemType>

    //CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: [[QUANT:%.+]] = IE.Quantize([[SUBTRACT]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: return [[QUANT]] : tensor<1x16x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @DoNotFuseMixedPrecisionAddPerAxisOut
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>, [[ARG1:%.+]]: tensor<1x16x1x1xf16>)
func.func @DoNotFuseMixedPrecisionAddPerAxisOut(%arg0: tensor<1x16x1x1xf16>, %arg1: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1x!qElemType> {
    %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    %quantize = IE.Quantize(%add) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>

    return %quantize : tensor<1x16x1x1x!qElemType>

    //CHECK: [[ADD:%.+]] = IE.Add([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: [[QUANT:%.+]] = IE.Quantize([[ADD]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: return [[QUANT]] : tensor<1x16x1x1x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @DoNotFuseMixedPrecisionAvgPoolPerAxisOut
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x3x3xf16>)
func.func @DoNotFuseMixedPrecisionAvgPoolPerAxisOut(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %avgPool = IE.AvgPool(%arg0) {
        kernel_size = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %quantize = IE.Quantize(%avgPool) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %quantize : tensor<1x16x3x3x!qElemType>

    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[ARG0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
    // CHECK: [[QUANT:%.+]] = IE.Quantize([[AVGPOOL]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    // CHECK: return [[QUANT]] : tensor<1x16x3x3x!qElemType>
}
