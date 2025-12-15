//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-weights-to-u8 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8<0:254>:f16:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>
!qElemType2 = !quant.uniform<u8:f16, 1.1534313725490195>
!qElemType3 = !quant.uniform<u8:f16, 2.4627450980392158>

func.func @Conv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType1> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    %2 = IE.Convolution(%1, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14x!qElemType3>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    return %3 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.*]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType1>,
    // CHECK-SAME:      #const.ConvertElemType<!qElemType>

    // CHECK:       [[VAL1:%.*]] = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    // CHECK:       [[VAL2:%.*]] = IE.Convolution([[VAL1]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType3>
    // CHECK:       [[VAL3:%.*]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8Constants
func.func @MixedPrecisionKeepI8Constants(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8Arguments
func.func @MixedPrecisionKeepI8Arguments(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!qElemType>) -> tensor<1x16x16x16xf16> {
  %result = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-LABEL: @MixedPrecisionI8Arguments
func.func @MixedPrecisionI8Arguments(%arg0: tensor<1x16x16x16x!qElemType>, %arg1: tensor<16x16x1x1x!qElemType>) -> tensor<1x16x16x16xf16> {
  %result = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL1:%.*]] = IE.Convolution([[U8_ARG0:%.*]], [[U8_ARG1:%.*]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType1>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8ConstantsMultipleConsumers
func.func @MixedPrecisionKeepI8ConstantsMultipleConsumers(%arg0: tensor<1x16x16x16xf16>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>) {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %result1 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  %result2 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result1, %result2 : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]], [[VAL2]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK-LABEL: @MixedPrecisionMultipleConsumers
func.func @MixedPrecisionMultipleConsumers(%arg0: tensor<1x16x16x16x!qElemType>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>) {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %result1 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  %result2 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result1, %result2 : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType1>
  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType1>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x16x16xf16>
  //CHECK: [[VAL2:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType1>, tensor<16x16x1x1x!qElemType1> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]], [[VAL2]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI4Constants
func.func @KeepI4Constants(%arg0: tensor<1x16x16x16x!qElemType>) -> tensor<1x16x16x16x!qElemType> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16x!qElemType>

  return %result : tensor<1x16x16x16x!qElemType>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI4Constants
func.func @MixedPrecisionKeepI4Constants(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.*]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 0.040252051633947038:118>
!qElemType2 = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType3 = !quant.uniform<u8:f16, 0.0080899796503431654:2>

// CHECK-LABEL: @MixedPrecisionAndNotMixedMultipleConsumers
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<8x64x1x1xf16>
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<8x64x1x1x!qElemType>
func.func @MixedPrecisionAndNotMixedMultipleConsumers(%arg0: tensor<8x64x1x1xf16>, %arg1: tensor<8x64x1x1x!qElemType>) -> (tensor<8x76x1x1x!qElemType1>, tensor<8x76x1x1xf16>) {
  %cst = const.Declare tensor<76x64x1x1x!qElemType2> = dense<2.000000e+00> : tensor<76x64x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType2>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<8x64x1x1xf16>, tensor<76x64x1x1x!qElemType2> -> tensor<8x76x1x1x!qElemType1>
  %1 = IE.Convolution(%arg1, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<8x64x1x1x!qElemType>, tensor<76x64x1x1x!qElemType2> -> tensor<8x76x1x1xf16>
  return %0, %1 : tensor<8x76x1x1x!qElemType1>, tensor<8x76x1x1xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<76x64x1x1x!qElemType2> = dense<2.000000e+00> : tensor<76x64x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType2>]
  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<76x64x1x1x!qElemType3> = dense<2.000000e+00> : tensor<76x64x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType2>, #const.ConvertElemType<!qElemType3>]
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[CST]])
  // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<8x64x1x1xf16>, tensor<76x64x1x1x!qElemType2>
  // CHECK-SAME:          -> tensor<8x76x1x1x!qElemType1>
  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[INPUT_0]], [[CST_0]])
  // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<8x64x1x1x!qElemType>, tensor<76x64x1x1x!qElemType3>
  // CHECK-SAME:          -> tensor<8x76x1x1xf16>

  // CHECK:       return [[CONV]], [[CONV_1]] : tensor<8x76x1x1x!qElemType1>, tensor<8x76x1x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i8:f16, 2.000000e+00>
// CHECK-DAG:  #[[MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<i8:f16, 2.000000e+00>

// CHECK:      func.func @MixedPrecisionSubgraphKeepI8Arguments
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<64x64x!qElemType>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @MixedPrecisionSubgraphKeepI8Arguments(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x!qElemType1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType1> -> tensor<64x64x!qElemType1>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<64x64x!qElemType1> -> tensor<64x64x!qElemType>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType> -> tensor<64x64x!qElemType>
  %3 = IE.Reshape(%2) {shape_value = [64, 64, 1]} : tensor<64x64x!qElemType> -> tensor<64x64x1x!qElemType>
  %4 = IE.AffineReshape(%3) { dim_mapping = [[0], [1], [2, 3]], shape_value = [64, 64, 1, 1] } : tensor<64x64x1x!qElemType> -> tensor<64x64x1x1x!qElemType>
  %5 = IE.Convolution(%arg0, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %5 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:               [[VAL0:%.*]] = IE.Transpose([[ARG1:%.*]]) {order_value = #[[MAP]]} : tensor<64x64x[[Q_ELEM_TYPE1]]> -> tensor<64x64x[[Q_ELEM_TYPE1]]>
  // CHECK:               [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x64x[[Q_ELEM_TYPE1]]> -> tensor<64x64x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL2:%.*]] = IE.Transpose([[VAL1]]) {order_value = #[[MAP]]} : tensor<64x64x[[Q_ELEM_TYPE0]]> -> tensor<64x64x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL3:%.*]] = IE.AffineReshape([[VAL2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [64, 64, 1, 1]}
  // CHECK-SAME:          tensor<64x64x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL4:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:               return [[VAL4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.*]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.*]] = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<i8:f16, 0.0080899796503431654:-126>

// CHECK:      func.func @MixedPrecisionSubgraphKeepI8ArgumentsWhenHasDiffQuantType
// CHECK-SAME: ([[ARG0:%.*]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.*]]: tensor<64x64x1x!qElemType>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @MixedPrecisionSubgraphKeepI8ArgumentsWhenHasDiffQuantType(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x!qElemType1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<64x64x1x!qElemType1> -> tensor<64x1x64x!qElemType1>
  %1 = IE.Reshape(%0) {shape_value = [64, 1, 64, 1]} : tensor<64x1x64x!qElemType1> -> tensor<64x1x64x1x!qElemType1>
  %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<64x1x64x1x!qElemType1> -> tensor<64x1x64x1x!qElemType>
  %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<64x1x64x1x!qElemType> -> tensor<64x64x1x1x!qElemType>
  %4 = IE.Convolution(%arg0, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %4 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:               [[VAL0:%.*]] = IE.AffineReshape([[ARG1:%.*]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [64, 1, 64, 1]}
  // CHECK-SAME:        : tensor<64x64x1x[[Q_ELEM_TYPE1]]> -> tensor<64x1x64x1x[[Q_ELEM_TYPE1]]>
  // CHECK:               [[VAL1:%.*]] = IE.QuantizeCast([[VAL0]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x1x64x1x[[Q_ELEM_TYPE1]]> -> tensor<64x1x64x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [64, 64, 1, 1]}
  // CHECK-SAME:        : tensor<64x1x64x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL3:%.*]] = IE.Convolution([[ARG0:%.*]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:               return [[VAL3]]
}

// -----

// CHECK:     !qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK:      func.func @MixedPrecisionSubgraphI8Arguments
// CHECK-SAME: ([[ARG0:%.*]]: tensor<2x32x64x100xsi8>, [[ARG1:%.*]]: tensor<64x64xsi8>) -> tensor<1x64x64x100xf16>
func.func @MixedPrecisionSubgraphI8Arguments(%arg0: tensor<2x32x64x100xsi8>, %arg1: tensor<64x64xsi8>) -> tensor<1x64x64x100xf16> {
  %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<64x64xsi8> -> tensor<64x64x!qElemType1>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType1> -> tensor<64x64x!qElemType1>
  %2 = IE.Reshape(%1) {shape_value = [64, 64, 1]} : tensor<64x64x!qElemType1> -> tensor<64x64x1x!qElemType1>
  %3 = IE.AffineReshape(%2) { dim_mapping = [[0], [1], [2, 3]], shape_value = [64, 64, 1, 1] } : tensor<64x64x1x!qElemType1> -> tensor<64x64x1x1x!qElemType1>
  %4 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<2x32x64x100xsi8> -> tensor<2x32x64x100x!qElemType1>
  %5 = IE.Reshape(%4) {shape_value = [1, 64, 64, 100]} : tensor<2x32x64x100x!qElemType1> -> tensor<1x64x64x100x!qElemType1>
  %6 = IE.Convolution(%5, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType1>, tensor<64x64x1x1x!qElemType1> -> tensor<1x64x64x100xf16>
  return %6 : tensor<1x64x64x100xf16>

  // CHECK:               [[VAL0:%.*]] = IE.QuantizeCast([[ARG1:%.*]]) {dstElemType = !qElemType} : tensor<64x64xsi8> -> tensor<64x64x!qElemType>
  // CHECK:               [[VAL1:%.*]] = IE.Transpose(%0) {order_value = #[[MAP]]} : tensor<64x64x!qElemType> -> tensor<64x64x!qElemType>
  // CHECK:               [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [64, 64, 1, 1]} : tensor<64x64x!qElemType> -> tensor<64x64x1x1x!qElemType>
  // CHECK:               [[VAL3:%.*]] = IE.QuantizeCast([[ARG0:%.*]]) {dstElemType = !qElemType} : tensor<2x32x64x100xsi8> -> tensor<2x32x64x100x!qElemType>
  // CHECK:               [[VAL4:%.*]] = IE.Reshape([[VAL3]]) {shape_value = [1, 64, 64, 100]} : tensor<2x32x64x100x!qElemType> -> tensor<1x64x64x100x!qElemType>
  // CHECK:               [[VAL5:%.*]] = IE.Convolution([[VAL4]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16>
  // CHECK:               return [[VAL5]]
}


// -----

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK:      func.func @SkipForQuantizeCastDequantizePattern
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1x3584xsi8>
func.func @SkipForQuantizeCastDequantizePattern(%arg0: tensor<1x1x1x3584xsi8>) -> tensor<1x1x1x3584xf16> {
  %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<1x1x1x3584xsi8> -> tensor<1x1x1x3584x!qElemType>
  %1 = IE.Dequantize(%0) {dstElemType = f16} : tensor<1x1x1x3584x!qElemType> -> tensor<1x1x1x3584xf16>

  return %1 : tensor<1x1x1x3584xf16>

  // CHECK:      [[QUANTIZECAST:%.+]] = IE.QuantizeCast([[INPUT]]) {dstElemType = !qElemType} : tensor<1x1x1x3584xsi8> -> tensor<1x1x1x3584x!qElemType>
  // CHECK:      [[DEQUANTIZE:%.+]] = IE.Dequantize([[QUANTIZECAST]]) {dstElemType = f16} : tensor<1x1x1x3584x!qElemType> -> tensor<1x1x1x3584xf16>
  // CHECK:      return [[DEQUANTIZE]] : tensor<1x1x1x3584xf16>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0472412109375>

// CHECK:      func.func @KeepFP16ActivationAndQuantWeightsAsI8WithMultipleSliceUsers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x256x1024xf16>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<1x1024x2x256xsi8>
func.func @KeepFP16ActivationAndQuantWeightsAsI8WithMultipleSliceUsers(%arg0: tensor<1x2x256x1024xf16>, %arg1: tensor<1x1024x2x256xsi8>) -> tensor<1x2x1024x1024xf16> {
  %0 = IE.Reshape(%arg0) {shape_value = [1, 2, 1024, 256]} : tensor<1x2x256x1024xf16> -> tensor<1x2x1024x256xf16>
  %1 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1x1024x2x256xsi8> -> tensor<1x1024x2x256x!qElemType>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x1024x2x256x!qElemType> -> tensor<1x2x1024x256x!qElemType>
  %3 = IE.Slice %0 [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256xf16> to tensor<1x1x1024x256xf16>
  %4 = IE.Slice %0 [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256xf16> to tensor<1x1x1024x256xf16>
  %5 = IE.Slice %2 [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  %6 = IE.Slice %2 [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  %7 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256xf16> -> tensor<1024x256x1x1xf16>
  %8 = IE.AffineReshape(%5) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  %9 = IE.Convolution(%7, %8) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>
  %10 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256xf16> -> tensor<1024x256x1x1xf16>
  %11 = IE.AffineReshape(%6) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  %12 = IE.Convolution(%10, %11) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>
  %13 = IE.AffineReshape(%9) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024x1x1xf16> -> tensor<1x1x1024x1024xf16>
  %14 = IE.AffineReshape(%12) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024x1x1xf16> -> tensor<1x1x1024x1024xf16>
  %15 = IE.Concat(%13, %14) {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x2x1024x1024xf16>

  return %15 : tensor<1x2x1024x1024xf16>

  // CHECK:       [[RESHAPED_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 2, 1024, 256]} : tensor<1x2x256x1024xf16> -> tensor<1x2x1024x256xf16>
  // CHECK:       [[QUANTIZED_WEIGHTS:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType} : tensor<1x1024x2x256xsi8> -> tensor<1x1024x2x256x!qElemType>
  // CHECK:       [[TRANSPOSED_WEIGHTS:%.+]] = IE.Transpose([[QUANTIZED_WEIGHTS]]) {order_value = #NHCW} : tensor<1x1024x2x256x!qElemType> -> tensor<1x2x1024x256x!qElemType>
  // CHECK:       [[SLICE_INPUT_0:%.+]] = IE.Slice [[RESHAPED_INPUT]] [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256xf16> to tensor<1x1x1024x256xf16>
  // CHECK:       [[SLICE_INPUT_1:%.+]] = IE.Slice [[RESHAPED_INPUT]] [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256xf16> to tensor<1x1x1024x256xf16>
  // CHECK:       [[SLICE_WEIGHTS_0:%.+]] = IE.Slice [[TRANSPOSED_WEIGHTS]] [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  // CHECK:       [[SLICE_WEIGHTS_1:%.+]] = IE.Slice [[TRANSPOSED_WEIGHTS]] [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  // CHECK:       [[RESHAPE_INPUT_0:%.+]] = IE.AffineReshape([[SLICE_INPUT_0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256xf16> -> tensor<1024x256x1x1xf16>
  // CHECK:       [[RESHAPE_WEIGHTS_0:%.+]] = IE.AffineReshape([[SLICE_WEIGHTS_0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_INPUT_0]], [[RESHAPE_WEIGHTS_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>

  // CHECK:       [[RESHAPE_INPUT_1:%.+]] = IE.AffineReshape([[SLICE_INPUT_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256xf16> -> tensor<1024x256x1x1xf16>
  // CHECK:       [[RESHAPE_WEIGHTS_1:%.+]] = IE.AffineReshape([[SLICE_WEIGHTS_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_INPUT_1]], [[RESHAPE_WEIGHTS_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>

  // CHECK:       [[RESHAPE_CONV_0:%.+]] = IE.AffineReshape([[CONV_0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024x1x1xf16> -> tensor<1x1x1024x1024xf16>
  // CHECK:       [[RESHAPE_CONV_1:%.+]] = IE.AffineReshape([[CONV_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1024, 1024]} : tensor<1024x1024x1x1xf16> -> tensor<1x1x1024x1024xf16>

  // CHECK:       [[RESULT:%.+]] = IE.Concat([[RESHAPE_CONV_0]], [[RESHAPE_CONV_1]])
  // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1024x1024xf16>, tensor<1x1x1024x1024xf16> -> tensor<1x2x1024x1024xf16>
  // CHECK:       return [[RESULT]]
}
