//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-to-mixed-precision %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionConv
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
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

  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<i8:f16, 1.1534313725490195:-1>

// CHECK-LABEL: @AvoidMixedPrecisionConvWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x1x1xf16>)
func.func @AvoidMixedPrecisionConvWithDifferentIntegerSignedness(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType1> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType1> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  // CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
  // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[DEQUANTIZE]], [[DEQUANTIZE1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  // CHECK: return [[CONV]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @AvoidMixedPrecisionConvPerAxes
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
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

  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
  //CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[VAL4:%.+]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionGroupConv
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionGroupConv(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<i8:f16, 2.000000e+00:-1>

// CHECK-LABEL: @AvoidMixedPrecisionGroupConvWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x3x3xf16>)
func.func @AvoidMixedPrecisionGroupConvWithDifferentIntegerSignedness(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    // CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>
    // CHECK: [[GCONV:%.+]] = IE.GroupConvolution([[DEQUANTIZE]], [[DEQUANTIZE1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>
    // CHECK: return [[GCONV]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @MixedPrecisionGroupConvPerAxes
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionGroupConvPerAxes(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType1>, tensor<16x1x3x3x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAdd
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @MixedPrecisionAdd(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Add([[VAL0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1x!qElemType>, tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @AvoidMixedPrecisionAddDifferentOperandType
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AvoidMixedPrecisionAddDifferentOperandType(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL2:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    //CHECK: [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>

// CHECK-LABEL: @AvoidMixedPrecisionAddPerAxes
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AvoidMixedPrecisionAddPerAxes(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<1x16x1x1x!qElemType> = dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x1x1x!qElemType> -> tensor<1x16x1x1xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType>
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
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

  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x3x3xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionGroupConvForOutputShape
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionGroupConvForOutputShape(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<16x1x1x1x!qElemType> = dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x1x1x!qElemType> -> tensor<16x1x1x1xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x1x1x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3x!qElemType>, tensor<16x1x1x1x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL1]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAvgPoolForOutputShape
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x64x88x88x!qElemType>
func.func @MixedPrecisionAvgPoolForOutputShape(%arg0: tensor<1x64x88x88x!qElemType>) -> tensor<1x64x11x11xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x88x88xf16>
    %1 = IE.AvgPool(%0) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88xf16> -> tensor<1x64x11x11xf16>

    return %1 : tensor<1x64x11x11xf16>

    // CHECK-NOT: IE.Dequantize
    // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[ARG_0]]) {exclude_pads, kernel_size = [8, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} : tensor<1x64x88x88x!qElemType> -> tensor<1x64x11x11xf16>

}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionTransposedConvForOutputShape
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x20x23x30xf16>
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

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x20x23x30xf16> -> tensor<1x20x23x30x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.TransposedConvolution([[VAL0]], [[CST]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x20x23x30x!qElemType>, tensor<20x20x2x2x!qElemType> -> tensor<1x20x46x60xf16>

    //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00:2>

// CHECK-LABEL: @AvoidMixedPrecisionTransposedConvForOutputShapeWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x20x23x30xf16>)
func.func @AvoidMixedPrecisionTransposedConvForOutputShapeWithDifferentIntegerSignedness(%arg0: tensor<1x20x23x30xf16>) -> tensor<1x20x46x60xf16> {
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x20x23x30xf16> -> tensor<1x20x23x30x!qElemType>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x20x23x30x!qElemType> -> tensor<1x20x23x30xf16>
    %weights = const.Declare tensor<20x20x2x2x!qElemType1> = dense<1.0> : tensor<20x20x2x2xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<20x20x2x2x!qElemType1> -> tensor<20x20x2x2xf16>
    %4 = IE.TransposedConvolution(%2, %3) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x20x23x30xf16>, tensor<20x20x2x2xf16> -> tensor<1x20x46x60xf16>
    return %4 : tensor<1x20x46x60xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<20x20x2x2x!qElemType> =
    // CHECK-SAME:   dense<1.000000e+00> : tensor<20x20x2x2xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x20x23x30xf16> -> tensor<1x20x23x30x!qElemType1>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x20x23x30x!qElemType1> -> tensor<1x20x23x30xf16>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<20x20x2x2x!qElemType> -> tensor<20x20x2x2xf16>
    // CHECK: [[TCONV:%.+]] = IE.TransposedConvolution([[DEQUANTIZE]], [[DEQUANTIZE1]]) {dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]} : tensor<1x20x23x30xf16>, tensor<20x20x2x2xf16> -> tensor<1x20x46x60xf16>
    // CHECK: return [[TCONV]] : tensor<1x20x46x60xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionAddForSameScales
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @MixedPrecisionAddForSameScales(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
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

!qElemType = !quant.uniform<u8:f16, 0.0025215686274509803>

// CHECK-LABEL: @Conv2dWithQuantize
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.GroupConvolution([[ARG_0]], [[CST]]) {
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.AvgPool([[ARG_0]]) {
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x1x64x128xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.AvgPool([[ARG_0]]) {
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

// CHECK-LABEL: @DoNotFuseQuantParamsIntoAvgPoolWithExcludePadsAttr
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x3x135x240xf16>
func.func @DoNotFuseQuantParamsIntoAvgPoolWithExcludePadsAttr(%arg0: tensor<1x3x135x240xf16> ) -> tensor<1x3x68x120x!qElemType> {
    %0 = IE.AvgPool(%arg0) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType>
    return %1 : tensor<1x3x68x120x!qElemType>

    //CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[ARG_0]]) {exclude_pads, kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x3x135x240xf16> -> tensor<1x3x68x120xf16>
    // CHECK: [[RESULT:%.+]] = IE.Quantize([[AVGPOOL]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      } : tensor<1x3x68x120xf16> -> tensor<1x3x68x120x!qElemType>
    // CHECK: return [[RESULT]] : tensor<1x3x68x120x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @AddWithQuantize
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @AddWithQuantize(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %0 = IE.Add(%arg0, %arg0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK:   [[VAL0:%.+]] = IE.Add([[ARG_0]], [[ARG_0]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[VAL0]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.003937007874015748>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @MixedPrecisionGroupConvForOutputShapeWithQuantWeightsBias
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x3x320x480xf16>
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
    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x3x320x480xf16> -> tensor<1x3x320x480x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[CST0]]) {dstElemType = f16} : tensor<1x3x1x1x!qElemType> -> tensor<1x3x1x1xf16>
    //CHECK: [[VAL2:%.+]] = IE.GroupConvolution([[VAL0]], [[CST]], [[VAL1]]) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x320x480x!qElemType1>, tensor<3x1x1x1x!qElemType1>, tensor<1x3x1x1xf16> -> tensor<1x3x320x480xf16>
    //CHECK: return [[VAL2]] : tensor<1x3x320x480xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>

// CHECK-LABEL: @DoNotAddWithQuantize
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x8x8x32xf16>
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
    // CHECK:   [[VAL0:%.+]] = IE.Add([[ARG_0]], [[CST0]]) {
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
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
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

    // CHECK:   [[VAL0:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
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

// CHECK-LABEL: @AvoidMixedPrecisionForInvalidApproximation
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x1x256x256xf16>
func.func @AvoidMixedPrecisionForInvalidApproximation(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256x!qElemType> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 1 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
        strides = [1, 1]
    } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    return %1 : tensor<1x1x256x256x!qElemType>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    //CHECK-SAME:               dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    //CHECK: [[VAL0:%.+]] = IE.GroupConvolution([[ARG_0]], [[CST]]) {
    //CHECK-SAME:               clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
    //CHECK-SAME:               dilations = [1, 1],
    //CHECK-SAME:               groups = 1 : i64,
    //CHECK-SAME:               pads_begin = [0, 0],
    //CHECK-SAME:               pads_end = [0, 0],
    //CHECK-SAME:               strides = [1, 1]
    //CHECK-SAME:           } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    //CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    //CHECK: return [[VAL1]] : tensor<1x1x256x256x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 3.1368405211205576E-7>

// CHECK-LABEL: @AvoidMixedPrecisionForInvalidApproximationWithClamp
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x1x256x256xf16>
func.func @AvoidMixedPrecisionForInvalidApproximationWithClamp(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256x!qElemType> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.GroupConvolution(%arg0, %cst) {
        clamp = {min = 0.000000e+00 : f64, max = 1.000000e+00 : f64},
        dilations = [1, 1],
        groups = 1 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    %1 = IE.Quantize(%0) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    return %1 : tensor<1x1x256x256x!qElemType>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> =
    //CHECK-SAME:               dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    //CHECK: [[VAL0:%.+]] = IE.GroupConvolution([[ARG_0]], [[CST]]) {
    //CHECK-SAME:               clamp = {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64},
    //CHECK-SAME:               dilations = [1, 1],
    //CHECK-SAME:               groups = 1 : i64,
    //CHECK-SAME:               pads_begin = [0, 0],
    //CHECK-SAME:               pads_end = [0, 0],
    //CHECK-SAME:               strides = [1, 1]
    //CHECK-SAME:           } : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    //CHECK: [[VAL1:%.+]] = IE.Quantize([[VAL0]]) {dstElemType = !qElemType} : tensor<1x1x256x256xf16> -> tensor<1x1x256x256x!qElemType>

    //CHECK: return [[VAL1]] : tensor<1x1x256x256x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>

// CHECK-LABEL: @MixedPrecisionGroupMatmul
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x8x1x64xf16>
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

  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType>
  //CHECK: [[VAL2:%.+]] = IE.MatMul([[VAL1]], [[VAL0]]) {transpose_b} : tensor<1x8x1x64x!qElemType>, tensor<1x8x128x64x!qElemType> -> tensor<1x8x1x128xf16>
  //CHECK: return [[VAL2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<i8:f16, 3.1368405211205576E-7:-126>

// CHECK-LABEL: @AvoidMixedPrecisionGroupMatmulWithDifferentIntegerSignedness
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x1x64xf16>)
func.func @AvoidMixedPrecisionGroupMatmulWithDifferentIntegerSignedness(%arg0: tensor<1x8x1x64xf16>) -> tensor<1x8x1x128xf16> {
    %weights = const.Declare tensor<1x8x128x64x!qElemType> = dense<1.0> : tensor<1x8x128x64xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType1>
    %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x1x64x!qElemType1> -> tensor<1x8x1x64xf16>
    %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x128x64x!qElemType> -> tensor<1x8x128x64xf16>
    %4 = IE.MatMul(%2, %3) {transpose_b} : tensor<1x8x1x64xf16>, tensor<1x8x128x64xf16> -> tensor<1x8x1x128xf16>

    return %4 : tensor<1x8x1x128xf16>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<1x8x128x64x!qElemType> =
    // CHECK-SAME:   dense<1.000000e+00> : tensor<1x8x128x64xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    // CHECK: [[QUANTIZE:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType1} : tensor<1x8x1x64xf16> -> tensor<1x8x1x64x!qElemType1>
    // CHECK: [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZE]]) {dstElemType = f16} : tensor<1x8x1x64x!qElemType1> -> tensor<1x8x1x64xf16>
    // CHECK: [[DEQUANTIZE1:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<1x8x128x64x!qElemType> -> tensor<1x8x128x64xf16>
    // CHECK: [[MATMUL:%.+]] = IE.MatMul([[DEQUANTIZE]], [[DEQUANTIZE1]]) {transpose_b} : tensor<1x8x1x64xf16>, tensor<1x8x128x64xf16> -> tensor<1x8x1x128xf16>
    // CHECK: return [[MATMUL]] : tensor<1x8x1x128xf16>
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

// -----

!qElemType = !quant.uniform<u8:f16, -0.0039215686274509803:255>

// CHECK-LABEL: @DoNotFuseQuantizeWithNegativeScaleIntoConvWithReLU
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @DoNotFuseQuantizeWithNegativeScaleIntoConvWithReLU(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3x!qElemType> {
    %cst = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> : tensor<16x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        post_op = #IE.Relu<>,
        strides = [1, 1]
    } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.Quantize(%0) {
        dstElemType = !qElemType
    } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    return %1 : tensor<1x16x3x3x!qElemType>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<2.000000e+00> :
    // CHECK-SAME:  tensor<16x16x1x1xf16>

    // CHECK:   [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[CST]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      post_op = #IE.Relu<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME: } : tensor<1x16x3x3xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x3x3xf16>

    // CHECK:   [[QUANT:%.+]] = IE.Quantize([[CONV]]) {
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:  } : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>

    // CHECK:   return [[QUANT]] : tensor<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @AvoidMixedPrecisionConvWithIllegalQuantAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x1x1xf16>
func.func @AvoidMixedPrecisionConvWithIllegalQuantAxis(%arg0: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
  %weights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  %4 = IE.Convolution(%2, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>

  return %4 : tensor<1x16x1x1xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<16x16x1x1xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x16x1x1xf16> -> tensor<1x16x1x1x!qElemType1>
  //CHECK: [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x16x1x1x!qElemType1> -> tensor<1x16x1x1xf16>
  //CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<16x16x1x1x!qElemType> -> tensor<16x16x1x1xf16>
  //CHECK: [[VAL4:%.+]] = IE.Convolution([[VAL2]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x1xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x1x1xf16>
  //CHECK: return [[VAL4]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType1 = !quant.uniform<u8:f16:2, {1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128}>

// CHECK-LABEL: @AvoidMixedPrecisionGroupConvWithIllegalQuantAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @AvoidMixedPrecisionGroupConvWithIllegalQuantAxis(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x1x1xf16> {
    %cst = const.Declare tensor<16x1x3x3x!qElemType> = dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>

    %3 = IE.GroupConvolution(%1, %2) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>

    return %3 : tensor<1x16x1x1xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<16x1x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<16x1x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType1>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType1> -> tensor<1x16x3x3xf16>
    //CHECK: [[VAL2:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<16x1x3x3x!qElemType> -> tensor<16x1x3x3xf16>
    //CHECK: [[VAL3:%.+]] = IE.GroupConvolution([[VAL1]], [[VAL2]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x3x3xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x1x1xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x1x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:2, {1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128}>

// CHECK-LABEL: @AvoidMixedPrecisionAddWithIllegalQuantAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @AvoidMixedPrecisionAddWithIllegalQuantAxis(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %cst = const.Declare tensor<1x16x3x3x!qElemType> = dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]

    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>

    %3 = IE.Add(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %3 : tensor<1x16x3x3xf16>

    //CHECK: [[CST:%.+]] = const.Declare tensor<1x16x3x3x!qElemType> =
    //CHECK-SAME:     dense<2.000000e+00> : tensor<1x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: [[VAL2:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: [[VAL3:%.+]] = IE.Add([[VAL1]], [[VAL2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf16>, tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL3]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:2, {1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128}>

// CHECK-LABEL: @AvoidMixedPrecisionAvgPoolWithIllegalQuantAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x3x3xf16>
func.func @AvoidMixedPrecisionAvgPoolWithIllegalQuantAxis(%arg0: tensor<1x16x3x3xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    %2 = IE.AvgPool(%1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %2 : tensor<1x16x3x3xf16>

    //CHECK: [[VAL0:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3x!qElemType>
    //CHECK: [[VAL1:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x16x3x3x!qElemType> -> tensor<1x16x3x3xf16>
    //CHECK: [[VAL2:%.+]] = IE.AvgPool([[VAL1]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>
    //CHECK: return [[VAL2]] : tensor<1x16x3x3xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128, 1.000000e+00:128}>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @AvoidMixedPrecisionMatMulWithIllegalQuantAxis
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x8x1x16xf16>
func.func @AvoidMixedPrecisionMatMulWithIllegalQuantAxis(%arg0: tensor<1x8x1x16xf16>) -> tensor<1x8x1x128xf16> {
  %1 = IE.Quantize(%arg0) {dstElemType = !qElemType1} : tensor<1x8x1x16xf16> -> tensor<1x8x1x16x!qElemType1>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<1x8x1x16x!qElemType1> -> tensor<1x8x1x16xf16>
  %weights = const.Declare tensor<1x8x128x16x!qElemType> = dense<1.0> : tensor<1x8x128x16xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %3 = IE.Dequantize(%weights) {dstElemType = f16} : tensor<1x8x128x16x!qElemType> -> tensor<1x8x128x16xf16>
  %4 = IE.MatMul(%2, %3) {transpose_b} : tensor<1x8x1x16xf16>, tensor<1x8x128x16xf16> -> tensor<1x8x1x128xf16>

  return %4 : tensor<1x8x1x128xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<1x8x128x16x!qElemType> =
  //CHECK-SAME:                 dense<1.000000e+00> : tensor<1x8x128x16xf16>,
  //CHECK-SAME:                 [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  //CHECK: [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType1} : tensor<1x8x1x16xf16> -> tensor<1x8x1x16x!qElemType1>
  //CHECK: [[VAL2:%.+]] = IE.Dequantize([[VAL1]]) {dstElemType = f16} : tensor<1x8x1x16x!qElemType1> -> tensor<1x8x1x16xf16>
  //CHECK: [[VAL3:%.+]] = IE.Dequantize([[VAL0]]) {dstElemType = f16} : tensor<1x8x128x16x!qElemType> -> tensor<1x8x128x16xf16>
  //CHECK: [[VAL4:%.+]] = IE.MatMul([[VAL2]], [[VAL3]]) {transpose_b} : tensor<1x8x1x16xf16>, tensor<1x8x128x16xf16> -> tensor<1x8x1x128xf16>
  //CHECK: return [[VAL4]] : tensor<1x8x1x128xf16>
}
