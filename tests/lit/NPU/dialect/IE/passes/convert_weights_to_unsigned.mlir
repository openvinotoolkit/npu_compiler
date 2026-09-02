//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-weights-to-unsigned --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

!qElemType = !quant.uniform<u8<0:254>:f16:0, {0.010680671751968504:127,0.0081200787401574797:127,0.010596087598425197:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {0.010680671751968504,0.0081200787401574797,0.010596087598425197}>
!qElemType2 = !quant.uniform<u8:f16, 1.1534313725490195>
!qElemType3 = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @Conv
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x3x16x16xf16>
func.func @Conv(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x14x14xf16> {
    %0 = const.Declare tensor<3x3x3x3x!qElemType1> =
        dense<-1.0> : tensor<3x3x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType1>]
    %1 = IE.Quantize(%arg0) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    %2 = IE.Convolution(%1, %0) {
        dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]
    } : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType1> -> tensor<1x3x14x14x!qElemType3>
    %3 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    return %3 : tensor<1x3x14x14xf16>

    // CHECK:       [[VAL0:%.+]] = const.Declare tensor<3x3x3x3x!qElemType> =
    // CHECK-SAME:      dense<-1.000000e+00> : tensor<3x3x3x3xf16>,
    // CHECK-SAME:      #const.CastElemType<si8>,
    // CHECK-SAME:      #const.CastElemType<!qElemType1>,
    // CHECK-SAME:      #const.ConvertElemType<!qElemType>

    // CHECK:       [[VAL1:%.+]] = IE.Quantize([[ARG_0]]) {dstElemType = !qElemType2} : tensor<1x3x16x16xf16> -> tensor<1x3x16x16x!qElemType2>
    // CHECK:       [[VAL2:%.+]] = IE.Convolution([[VAL1]], [[VAL0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x3x16x16x!qElemType2>, tensor<3x3x3x3x!qElemType> -> tensor<1x3x14x14x!qElemType3>
    // CHECK:       [[VAL3:%.+]] = IE.Dequantize([[VAL2]]) {dstElemType = f16} : tensor<1x3x14x14x!qElemType3> -> tensor<1x3x14x14xf16>
    // CHECK:       return [[VAL3]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8Constants
// CHECK-SAME:     [[ARG_0:%.+]]: tensor<1x16x16x16xf16>
func.func @MixedPrecisionKeepI8Constants(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8Arguments
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<16x16x1x1x!qElemType>
func.func @MixedPrecisionKeepI8Arguments(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<16x16x1x1x!qElemType>) -> tensor<1x16x16x16xf16> {
  %result = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK: !qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK-LABEL: @SamePrecisionKeepI8Arguments
// CHECK-SAME:     [[ARG_0:%.+]]: tensor<1x16x16x16x!qElemType>
// CHECK-SAME:     [[ARG_1:%.+]]: tensor<16x16x1x1x!qElemType>
func.func @SamePrecisionKeepI8Arguments(%arg0: tensor<1x16x16x16x!qElemType>, %arg1: tensor<16x16x1x1x!qElemType>) -> tensor<1x16x16x16xf16> {
  %result = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI8ConstantsMultipleConsumers
// CHECK-SAME:     [[ARG_0:%.+]]: tensor<1x16x16x16xf16>
func.func @MixedPrecisionKeepI8ConstantsMultipleConsumers(%arg0: tensor<1x16x16x16xf16>) -> (tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>) {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %result1 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  %result2 = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result1, %result2 : tensor<1x16x16x16xf16>, tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: [[VAL2:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]], [[VAL2]]
}


// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @KeepI4Constants
// CHECK-SAME:     [[ARG_0:%.+]]: tensor<1x16x16x16x!qElemType>
func.func @KeepI4Constants(%arg0: tensor<1x16x16x16x!qElemType>) -> tensor<1x16x16x16x!qElemType> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16x!qElemType>

  return %result : tensor<1x16x16x16x!qElemType>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16x!qElemType>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16x!qElemType>
  //CHECK: return [[VAL1]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>
// CHECK-LABEL: @MixedPrecisionKeepI4Constants
// CHECK-SAME:     [[ARG_0:%.+]]: tensor<1x16x16x16xf16>
func.func @MixedPrecisionKeepI4Constants(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xf16> {
  %qweights = const.Declare tensor<16x16x1x1x!qElemType> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %result = IE.Convolution(%arg0, %qweights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>

  return %result : tensor<1x16x16x16xf16>

  //CHECK: [[VAL0:%.+]] = const.Declare tensor<16x16x1x1x!qElemType>
  //CHECK: [[VAL1:%.+]] = IE.Convolution([[ARG_0]], [[VAL0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1x!qElemType> -> tensor<1x16x16x16xf16>
  //CHECK: return [[VAL1]]
}

// -----

// CHECK: !qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
// CHECK: !qElemType1 = !quant.uniform<u8:f16, 0.040252051633947038:118>
// CHECK: !qElemType2 = !quant.uniform<i8:f16, 0.0080899796503431654>
// CHECK: !qElemType3 = !quant.uniform<u8:f16, 0.0080899796503431654:128>
!qElemType = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType1 = !quant.uniform<u8:f16, 0.040252051633947038:118>
!qElemType2 = !quant.uniform<i8:f16, 0.0080899796503431654>
!qElemType3 = !quant.uniform<u8:f16, 0.0080899796503431654:128>

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

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i8:f16, 2.000000e+00>
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<i8:f16, 2.000000e+00>

// CHECK:      func.func @MixedPrecisionSubgraphKeepI8Arguments
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<64x64x!qElemType>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @MixedPrecisionSubgraphKeepI8Arguments(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x!qElemType1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType1> -> tensor<64x64x!qElemType1>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<64x64x!qElemType1> -> tensor<64x64x!qElemType>
  %2 = IE.Transpose(%1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType> -> tensor<64x64x!qElemType>
  %3 = IE.Reshape(%2) {shape_value = [64, 64, 1]} : tensor<64x64x!qElemType> -> tensor<64x64x1x!qElemType>
  %4 = IE.AffineReshape(%3) { dim_mapping = [[0], [1], [2, 3]], shape_value = [64, 64, 1, 1] } : tensor<64x64x1x!qElemType> -> tensor<64x64x1x1x!qElemType>
  %5 = IE.Convolution(%arg0, %4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %5 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:               [[VAL0:%.+]] = IE.Transpose([[ARG1]]) {order_value = #[[MAP]]} : tensor<64x64x[[Q_ELEM_TYPE1]]> -> tensor<64x64x[[Q_ELEM_TYPE1]]>
  // CHECK:               [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x64x[[Q_ELEM_TYPE1]]> -> tensor<64x64x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL2:%.+]] = IE.Transpose([[VAL1]]) {order_value = #[[MAP]]} : tensor<64x64x[[Q_ELEM_TYPE0]]> -> tensor<64x64x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL3:%.+]] = IE.AffineReshape([[VAL2]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [64, 64, 1, 1]}
  // CHECK-SAME:          tensor<64x64x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL4:%.+]] = IE.Convolution([[ARG0]], [[VAL3]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:               return [[VAL4]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-DAG:  [[Q_ELEM_TYPE0:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG:  [[Q_ELEM_TYPE1:!.+]] = !quant.uniform<i8:f16, 0.0080899796503431654:-126>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<i8:f16, 0.0080899796503431654:-126>

// CHECK:      func.func @MixedPrecisionSubgraphKeepI8ArgumentsWhenHasDiffQuantType
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x64x100xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<64x64x1x!qElemType>) -> tensor<1x64x64x100xf16, {order = #NHWC}>
func.func @MixedPrecisionSubgraphKeepI8ArgumentsWhenHasDiffQuantType(%arg0: tensor<1x64x64x100xf16, {order = #NHWC}>, %arg1: tensor<64x64x1x!qElemType1>) -> tensor<1x64x64x100xf16, {order = #NHWC}> {
  %0 = IE.Transpose(%arg1) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<64x64x1x!qElemType1> -> tensor<64x1x64x!qElemType1>
  %1 = IE.Reshape(%0) {shape_value = [64, 1, 64, 1]} : tensor<64x1x64x!qElemType1> -> tensor<64x1x64x1x!qElemType1>
  %2 = IE.QuantizeCast(%1) {dstElemType = !qElemType} : tensor<64x1x64x1x!qElemType1> -> tensor<64x1x64x1x!qElemType>
  %3 = IE.Transpose(%2) {order_value = #NHWC} : tensor<64x1x64x1x!qElemType> -> tensor<64x64x1x1x!qElemType>
  %4 = IE.Convolution(%arg0, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  return %4 : tensor<1x64x64x100xf16, {order = #NHWC}>

  // CHECK:               [[VAL0:%.+]] = IE.AffineReshape([[ARG1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [64, 1, 64, 1]}
  // CHECK-SAME:        : tensor<64x64x1x[[Q_ELEM_TYPE1]]> -> tensor<64x1x64x1x[[Q_ELEM_TYPE1]]>
  // CHECK:               [[VAL1:%.+]] = IE.QuantizeCast([[VAL0]]) {dstElemType = [[Q_ELEM_TYPE0]]} : tensor<64x1x64x1x[[Q_ELEM_TYPE1]]> -> tensor<64x1x64x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL2:%.+]] = IE.AffineReshape([[VAL1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [64, 64, 1, 1]}
  // CHECK-SAME:        : tensor<64x1x64x1x[[Q_ELEM_TYPE0]]> -> tensor<64x64x1x1x[[Q_ELEM_TYPE0]]>
  // CHECK:               [[VAL3:%.+]] = IE.Convolution([[ARG0]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}>, tensor<64x64x1x1x[[Q_ELEM_TYPE0]]> -> tensor<1x64x64x100xf16, {order = #NHWC}>
  // CHECK:               return [[VAL3]]
}

// -----

// CHECK-DAG: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK:      func.func @KeepI8ForI8Arguments
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x32x64x100xsi8>, [[ARG1:%.+]]: tensor<64x64xsi8>) -> tensor<1x64x64x100xf16>
func.func @KeepI8ForI8Arguments(%arg0: tensor<2x32x64x100xsi8>, %arg1: tensor<64x64xsi8>) -> tensor<1x64x64x100xf16> {
  %0 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<64x64xsi8> -> tensor<64x64x!qElemType>
  %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<64x64x!qElemType> -> tensor<64x64x!qElemType>
  %2 = IE.Reshape(%1) {shape_value = [64, 64, 1]} : tensor<64x64x!qElemType> -> tensor<64x64x1x!qElemType>
  %3 = IE.AffineReshape(%2) { dim_mapping = [[0], [1], [2, 3]], shape_value = [64, 64, 1, 1] } : tensor<64x64x1x!qElemType> -> tensor<64x64x1x1x!qElemType>
  %4 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<2x32x64x100xsi8> -> tensor<2x32x64x100x!qElemType>
  %5 = IE.Reshape(%4) {shape_value = [1, 64, 64, 100]} : tensor<2x32x64x100x!qElemType> -> tensor<1x64x64x100x!qElemType>
  %6 = IE.Convolution(%5, %3) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16>
  return %6 : tensor<1x64x64x100xf16>

  // CHECK:               [[VAL0:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<64x64xsi8> -> tensor<64x64x!qElemType>
  // CHECK:               [[VAL1:%.+]] = IE.Transpose([[VAL0]]) {order_value = #[[MAP]]} : tensor<64x64x!qElemType> -> tensor<64x64x!qElemType>
  // CHECK:               [[VAL2:%.+]] = IE.AffineReshape([[VAL1]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2, 3]], shape_value = [64, 64, 1, 1]} : tensor<64x64x!qElemType> -> tensor<64x64x1x1x!qElemType>
  // CHECK:               [[VAL3:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = !qElemType} : tensor<2x32x64x100xsi8> -> tensor<2x32x64x100x!qElemType>
  // CHECK:               [[VAL4:%.+]] = IE.Reshape([[VAL3]]) {shape_value = [1, 64, 64, 100]} : tensor<2x32x64x100x!qElemType> -> tensor<1x64x64x100x!qElemType>
  // CHECK:               [[VAL5:%.+]] = IE.Convolution([[VAL4]], [[VAL2]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100x!qElemType>, tensor<64x64x1x1x!qElemType> -> tensor<1x64x64x100xf16>
  // CHECK:               return [[VAL5]]
}

// -----

// CHECK:     !qElemType = !quant.uniform<i8:f16, 7.2735629510134459E-4>
!qElemType = !quant.uniform<i8:f16, 7.2735629510134459E-4>

// CHECK:      func.func @KeepI8ConvWithDequantizeInputsArguments
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3072x1x1xf16>, [[ARG1:%.+]]: tensor<3072x32xsi8>) -> tensor<1x32x1x1xf16>
func.func @KeepI8ConvWithDequantizeInputsArguments(%arg0: tensor<1x3072x1x1xf16>, %arg1: tensor<3072x32xsi8>) -> tensor<1x32x1x1xf16> {
  %quantCast = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<3072x32xsi8> -> tensor<3072x32x!qElemType>
  %afreshape0 = IE.AffineReshape(%quantCast) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3072, 1, 32]} : tensor<3072x32x!qElemType> -> tensor<1x3072x1x32x!qElemType>
  %transpose = IE.Transpose(%afreshape0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x3072x1x32x!qElemType> -> tensor<1x32x1x3072x!qElemType>
  %afreshape1 = IE.AffineReshape(%transpose) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32, 3072, 1, 1]} : tensor<1x32x1x3072x!qElemType> -> tensor<32x3072x1x1x!qElemType>
  %dequant = IE.Dequantize(%afreshape1) {dstElemType = f16} : tensor<32x3072x1x1x!qElemType> -> tensor<32x3072x1x1xf16>
  %conv = IE.Convolution(%arg0, %dequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3072x1x1xf16>, tensor<32x3072x1x1xf16> -> tensor<1x32x1x1xf16>
  return  %conv : tensor<1x32x1x1xf16>

  // CHECK:               [[QCAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<3072x32xsi8> -> tensor<3072x32x!qElemType>
  // CHECK:               [[AFRESHAPE:%.+]] = IE.AffineReshape([[QCAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3072, 1, 32]} : tensor<3072x32x!qElemType> -> tensor<1x3072x1x32x!qElemType>
  // CHECK:               [[TRANSPOSE:%.+]] = IE.Transpose([[AFRESHAPE]]) {order_value = #NWHC} : tensor<1x3072x1x32x!qElemType> -> tensor<1x32x1x3072x!qElemType>
  // CHECK:               [[AFRESHAPE1:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32, 3072, 1, 1]} : tensor<1x32x1x3072x!qElemType> -> tensor<32x3072x1x1x!qElemType>
  // CHECK:               [[DEQUANTIZE:%.+]] = IE.Dequantize([[AFRESHAPE1]]) {dstElemType = f16} : tensor<32x3072x1x1x!qElemType> -> tensor<32x3072x1x1xf16>
  // CHECK:               [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANTIZE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3072x1x1xf16>, tensor<32x3072x1x1xf16> -> tensor<1x32x1x1xf16>
  // CHECK:               return [[CONV]] : tensor<1x32x1x1xf16>
}

// -----

// CHECK:     !qElemType = !quant.uniform<i8:f16, 7.2735629510134459E-4>
!qElemType = !quant.uniform<i8:f16, 7.2735629510134459E-4>

// CHECK:      func.func @KeepI8ConvWithDynamicDequantizeInputsArguments
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3072x1x1xf16>, [[ARG1:%.+]]: tensor<3072x32xsi8>, [[ARG2:%.+]]: tensor<1x3072x1x1xf16>) -> tensor<1x32x1x1xf16>
func.func @KeepI8ConvWithDynamicDequantizeInputsArguments(%arg0: tensor<1x3072x1x1xf16>, %arg1: tensor<3072x32xsi8>, %arg2: tensor<1x3072x1x1xf16>) -> tensor<1x32x1x1xf16> {
  %quantCast = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<3072x32xsi8> -> tensor<3072x32x!qElemType>
  %afreshape0 = IE.AffineReshape(%quantCast) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3072, 1, 32]} : tensor<3072x32x!qElemType> -> tensor<1x3072x1x32x!qElemType>
  %transpose = IE.Transpose(%afreshape0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>} : tensor<1x3072x1x32x!qElemType> -> tensor<1x32x1x3072x!qElemType>
  %afreshape1 = IE.AffineReshape(%transpose) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32, 3072, 1, 1]} : tensor<1x32x1x3072x!qElemType> -> tensor<32x3072x1x1x!qElemType>
  %dequant = IE.DynamicDequantize(%afreshape1, %arg2) {dstElemType = f16} : tensor<32x3072x1x1x!qElemType>, tensor<1x3072x1x1xf16> -> tensor<32x3072x1x1xf16>
  %conv = IE.Convolution(%arg0, %dequant) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3072x1x1xf16>, tensor<32x3072x1x1xf16> -> tensor<1x32x1x1xf16>
  return  %conv : tensor<1x32x1x1xf16>

  // CHECK:               [[QCAST:%.+]] = IE.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<3072x32xsi8> -> tensor<3072x32x!qElemType>
  // CHECK:               [[AFRESHAPE:%.+]] = IE.AffineReshape([[QCAST]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3072, 1, 32]} : tensor<3072x32x!qElemType> -> tensor<1x3072x1x32x!qElemType>
  // CHECK:               [[TRANSPOSE:%.+]] = IE.Transpose([[AFRESHAPE]]) {order_value = #NWHC} : tensor<1x3072x1x32x!qElemType> -> tensor<1x32x1x3072x!qElemType>
  // CHECK:               [[AFRESHAPE1:%.+]] = IE.AffineReshape([[TRANSPOSE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [32, 3072, 1, 1]} : tensor<1x32x1x3072x!qElemType> -> tensor<32x3072x1x1x!qElemType>
  // CHECK:               [[DEQUANTIZE:%.+]] = IE.DynamicDequantize([[AFRESHAPE1]], [[ARG2]]) {dstElemType = f16} : tensor<32x3072x1x1x!qElemType>, tensor<1x3072x1x1xf16> -> tensor<32x3072x1x1xf16>
  // CHECK:               [[CONV:%.+]] = IE.Convolution([[ARG0]], [[DEQUANTIZE]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3072x1x1xf16>, tensor<32x3072x1x1xf16> -> tensor<1x32x1x1xf16>
  // CHECK:               return [[CONV]] : tensor<1x32x1x1xf16>
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

// -----

!qElemType = !quant.uniform<i8:f16, 0.0472412109375>
// CHECK-DAG:  [[QELEMTYPE:!.+]] = !quant.uniform<i8:f16, 0.0472412109375>

// CHECK: func.func @KeepQuantizeCastAsI8WithResult
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>
func.func @KeepQuantizeCastAsI8WithResult(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x16x16x16xsi8> {
  %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
  %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16x!qElemType>
  %1 = IE.QuantizeCast(%0) {dstElemType = si8} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xsi8>
  return %1 : tensor<1x16x16x16xsi8>

  // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16x[[QELEMTYPE]]>
  // CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[CONV]]) {dstElemType = si8} : tensor<1x16x16x16x[[QELEMTYPE]]> -> tensor<1x16x16x16xsi8>
  // CHECK: return [[QCAST]] : tensor<1x16x16x16xsi8>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8:f16, 0.0472412109375>
// CHECK-DAG:  [[QELEMTYPE:!.+]] = !quant.uniform<i8:f16, 0.0472412109375>

// CHECK: func.func @KeepQuantizeCastAsI8WithTransposeSliceReshapeResult
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>
func.func @KeepQuantizeCastAsI8WithTransposeSliceReshapeResult(%arg0: tensor<1x16x16x16xf16>) -> tensor<1x256xsi8> {
  %weights = const.Declare tensor<16x16x1x1xf16> = dense<1.0> : tensor<16x16x1x1xf16>
  %0 = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16x!qElemType>
  %1 = IE.QuantizeCast(%0) {dstElemType = si8} : tensor<1x16x16x16x!qElemType> -> tensor<1x16x16x16xsi8>
  %2 = IE.Transpose(%1) {order_value = #NHWC} : tensor<1x16x16x16xsi8> -> tensor<1x16x16x16xsi8>
  %3 = IE.Slice %2 [0, 0, 0, 0] [1, 16, 1, 16] : tensor<1x16x16x16xsi8> to tensor<1x16x1x16xsi8>
  %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 256]} : tensor<1x16x1x16xsi8> -> tensor<1x256xsi8>
  return %4 : tensor<1x256xsi8>

  // CHECK: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
  // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG_0]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<16x16x1x1xf16> -> tensor<1x16x16x16x[[QELEMTYPE]]>
  // CHECK: [[QCAST:%.+]] = IE.QuantizeCast([[CONV]]) {dstElemType = si8} : tensor<1x16x16x16x[[QELEMTYPE]]> -> tensor<1x16x16x16xsi8>
  // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[QCAST]]) {order_value = #NHWC} : tensor<1x16x16x16xsi8> -> tensor<1x16x16x16xsi8>
  // CHECK: [[SLICE:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 16, 1, 16] : tensor<1x16x16x16xsi8> to tensor<1x16x1x16xsi8>
  // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[SLICE]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [1]], shape_value = [1, 256]} : tensor<1x16x1x16xsi8> -> tensor<1x256xsi8>
  // CHECK: return [[RESHAPE]] : tensor<1x256xsi8>
}

// -----

!qElemType = !quant.uniform<i8:f16, 0.0123456789>
// CHECK-DAG:  [[QELEMTYPE_I8:!.+]] = !quant.uniform<i8:f16, 0.0123456789>
// CHECK-DAG:  [[QELEMTYPE_U8:!.+]] = !quant.uniform<u8:f16, 0.0123456789:128>

// CHECK: func.func @ConvertConvWithFp16Results
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<32x16x3x3xf16>
func.func @ConvertConvWithFp16Results(%arg0: tensor<1x16x16x16xf16>, %arg1: tensor<32x16x3x3xf16>) -> tensor<1x32x12x12xf16> {
  %weights = const.Declare tensor<32x32x3x3x!qElemType> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %conv1 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x14x14x!qElemType>
  %conv2 = IE.Convolution(%conv1, %weights) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x14x14x!qElemType>, tensor<32x32x3x3x!qElemType> -> tensor<1x32x12x12xf16>
  return %conv2 : tensor<1x32x12x12xf16>

  // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<32x32x3x3x[[QELEMTYPE_U8]]> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.CastElemType<si8>, #const.CastElemType<[[QELEMTYPE_I8]]>, #const.ConvertElemType<[[QELEMTYPE_U8]]>]
  // CHECK:       [[CONV1:%.+]] = IE.Convolution([[ARG_0]], [[ARG_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x16x16xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x14x14x[[QELEMTYPE_U8]]>
  // CHECK:       [[CONV2:%.+]] = IE.Convolution([[CONV1]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x14x14x[[QELEMTYPE_U8]]>, tensor<32x32x3x3x[[QELEMTYPE_U8]]> -> tensor<1x32x12x12xf16>
  // CHECK:       return [[CONV2]] : tensor<1x32x12x12xf16>
}

// -----

// CHECK-DAG:     !qElemType = !quant.uniform<i8:f16, 0.047244105488061905>
!qElemType = !quant.uniform<i8:f16, 0.047244105488061905>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL:  @KeepI8ForConcatTransposeSliceConv
// CHECK-SAME:      [[INPUT_0:%arg[0-9]+]]: tensor<1x256x2x256xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]+]]: tensor<1x256x2x256xf16>
// CHECK-SAME:      [[WEIGHTS:%arg[0-9]+]]: tensor<1x768x2x256xsi8>
// CHECK-SAME:      [[ACT_0:%arg[0-9]+]]: tensor<1024x256x1x1xf16>
// CHECK-SAME:      [[ACT_1:%arg[0-9]+]]: tensor<1024x256x1x1xf16>
func.func @KeepI8ForConcatTransposeSliceConv(%arg0: tensor<1x256x2x256xf16>, %arg1: tensor<1x256x2x256xf16>, %arg2: tensor<1x768x2x256xsi8>, %arg3: tensor<1024x256x1x1xf16>, %arg4: tensor<1024x256x1x1xf16>) -> (tensor<1024x1024x1x1xf16>, tensor<1024x1024x1x1xf16>) {
  %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x256xf16>, tensor<1x256x2x256xf16> -> tensor<1x256x2x256x!qElemType>
  %quantize_cast_0 = IE.QuantizeCast(%add) {dstElemType = si8} : tensor<1x256x2x256x!qElemType> -> tensor<1x256x2x256xsi8>
  %concat = IE.Concat(%quantize_cast_0, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x2x256xsi8>, tensor<1x768x2x256xsi8> -> tensor<1x1024x2x256xsi8>
  %quantize_cast_1 = IE.QuantizeCast(%concat) {dstElemType = !qElemType} : tensor<1x1024x2x256xsi8> -> tensor<1x1024x2x256x!qElemType>
  %transpose = IE.Transpose(%quantize_cast_1) {order_value = #NHCW} : tensor<1x1024x2x256x!qElemType> -> tensor<1x2x1024x256x!qElemType>
  %slice_0 = IE.Slice %transpose [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  %slice_1 = IE.Slice %transpose [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  %reshape_0 = IE.AffineReshape(%slice_0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  %reshape_1 = IE.AffineReshape(%slice_1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  %conv_0 = IE.Convolution(%arg3, %reshape_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>
  %conv_1 = IE.Convolution(%arg4, %reshape_1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>

  return %conv_0, %conv_1 : tensor<1024x1024x1x1xf16>, tensor<1024x1024x1x1xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]])
  // CHECK-SAME:            {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x2x256xf16>, tensor<1x256x2x256xf16> -> tensor<1x256x2x256x!qElemType>
  // CHECK:       [[QUANTIZE_CAST_0:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = si8} : tensor<1x256x2x256x!qElemType> -> tensor<1x256x2x256xsi8>
  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[QUANTIZE_CAST_0]], [[WEIGHTS]])
  // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x2x256xsi8>, tensor<1x768x2x256xsi8> -> tensor<1x1024x2x256xsi8>
  // CHECK:       [[QUANTIZE_CAST_1:%.+]] = IE.QuantizeCast([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x1024x2x256xsi8> -> tensor<1x1024x2x256x!qElemType>
  // CHECK:       [[TRANSPOSE:%.+]] = IE.Transpose([[QUANTIZE_CAST_1]])
  // CHECK-SAME{LITERAL}:   {order_value = #NHCW} : tensor<1x1024x2x256x!qElemType> -> tensor<1x2x1024x256x!qElemType>
  // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[TRANSPOSE]] [0, 0, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[TRANSPOSE]] [0, 1, 0, 0] [1, 1, 1024, 256] : tensor<1x2x1024x256x!qElemType> to tensor<1x1x1024x256x!qElemType>
  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[SLICE_0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[SLICE_1]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 256, 1, 1]} : tensor<1x1x1024x256x!qElemType> -> tensor<1024x256x1x1x!qElemType>
  // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[ACT_0]], [[RESHAPE_0]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>
  // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[ACT_1]], [[RESHAPE_1]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x256x1x1xf16>, tensor<1024x256x1x1x!qElemType> -> tensor<1024x1024x1x1xf16>

  // CHECK:       return [[CONV_0]], [[CONV_1]] : tensor<1024x1024x1x1xf16>, tensor<1024x1024x1x1xf16>
}

// -----

// CHECK:     !qElemType = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK:     !qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>

!qElemType = !quant.uniform<u8:f16, 0.5>
!qElemType1 = !quant.uniform<i8:f16, 1.000000e+00>


// CHECK-LABEL: @MixedPrecisionKeepI8ConstantsNonConstWeights
// CHECK-SAME: ([[WEIGHTS:%.+]]: tensor<32x32x1x1xsi8>, [[INPUT:%.+]]: tensor<1x32x100x100x!qElemType>) -> tensor<1x32x100x100xf16>
func.func @MixedPrecisionKeepI8ConstantsNonConstWeights(%arg0: tensor<32x32x1x1xsi8>, %arg1: tensor<1x32x100x100x!qElemType>) -> tensor<1x32x100x100xf16> {
  %qcast = IE.QuantizeCast(%arg0) {dstElemType = !qElemType1} : tensor<32x32x1x1xsi8> -> tensor<32x32x1x1x!qElemType1>
  %conv = IE.Convolution(%arg1, %qcast) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x100x100x!qElemType>, tensor<32x32x1x1x!qElemType1> -> tensor<1x32x100x100xf16>
  return %conv : tensor<1x32x100x100xf16>
  // CHECK:       [[QCAST:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType1} : tensor<32x32x1x1xsi8> -> tensor<32x32x1x1x!qElemType1>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[QCAST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x100x100x!qElemType>, tensor<32x32x1x1x!qElemType1> -> tensor<1x32x100x100xf16>
  // CHECK:       return [[CONV]] : tensor<1x32x100x100xf16>
}

// -----

// CHECK:     !qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
// CHECK:     !qElemType1 = !quant.uniform<i8:f16, 0.14705899556477864>
// CHECK:     !qElemType2 = !quant.uniform<i8:f16, 1.000000e+00>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
!qElemType1 = !quant.uniform<i8:f16, 0.14705899556477864>
!qElemType2 = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @MixedPrecisionKeepI8ConstantsNonConvConsumers
// CHECK-SAME:      [[ARG0:%.+]]: tensor<32x1x1x3x3xsi8>
func.func @MixedPrecisionKeepI8ConstantsNonConvConsumers(%arg0: tensor<32x1x1x3x3xsi8>) -> tensor<32x1x3x3x!qElemType> {

  %afreshape = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 32, 3, 3]} : tensor<32x1x1x3x3xsi8> -> tensor<1x32x3x3xsi8>
  %qcast = IE.QuantizeCast(%afreshape) {dstElemType = !qElemType1} : tensor<1x32x3x3xsi8> -> tensor<1x32x3x3x!qElemType1>
  %afreshape1 = IE.AffineReshape(%qcast) {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [32, 1, 3, 3]} : tensor<1x32x3x3x!qElemType1> -> tensor<32x1x3x3x!qElemType1>
  %qcast1 = IE.QuantizeCast(%afreshape1) {dstElemType = !qElemType2} : tensor<32x1x3x3x!qElemType1> -> tensor<32x1x3x3x!qElemType2>
  %avgpool = IE.AvgPool(%qcast1) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<32x1x3x3x!qElemType2> -> tensor<32x1x3x3x!qElemType>
  return %avgpool : tensor<32x1x3x3x!qElemType>
  // CHECK:       [[AFRESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1], [2], [2], [2], [3]], shape_value = [1, 32, 3, 3]} : tensor<32x1x1x3x3xsi8> -> tensor<1x32x3x3xsi8>
  // CHECK:       [[QCAST:%.+]] = IE.QuantizeCast([[AFRESHAPE]]) {dstElemType = !qElemType1} : tensor<1x32x3x3xsi8> -> tensor<1x32x3x3x!qElemType1>
  // CHECK:       [[AFRESHAPE1:%.+]] = IE.AffineReshape([[QCAST]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1, 2], [3]], shape_value = [32, 1, 3, 3]} : tensor<1x32x3x3x!qElemType1> -> tensor<32x1x3x3x!qElemType1>
  // CHECK:       [[QCAST1:%.+]] = IE.QuantizeCast([[AFRESHAPE1]]) {dstElemType = !qElemType2} : tensor<32x1x3x3x!qElemType1> -> tensor<32x1x3x3x!qElemType2>
  // CHECK:       [[AVGPOOL:%.+]] = IE.AvgPool([[QCAST1]]) {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<32x1x3x3x!qElemType2> -> tensor<32x1x3x3x!qElemType>
  // CHECK:       return [[AVGPOOL]] : tensor<32x1x3x3x!qElemType>
}

// -----

// CHECK:     !qElemType = !quant.uniform<i8:f16:0, {5.4931640625E-4,6.7042553541707059E-4,5.74307172906165E-4,6.6369725208656461E-4}>
// CHECK:     !qElemType1 = !quant.uniform<u8:f16:0, {5.4931640625E-4:128,6.7042553541707059E-4:128,5.74307172906165E-4:128,6.6369725208656461E-4:128}>
// CHECK:     !qElemType2 = !quant.uniform<u8:f16, 3.1372548227070592E-7>

!qElemType = !quant.uniform<i8:f16:0, {5.4931640625E-4,6.7042553541707059E-4,5.74307172906165E-4,6.6369725208656461E-4}>
!qElemType1 = !quant.uniform<u8:f16, 3.1372548227070592E-7>

// CHECK-LABEL: @SkipFullQuantizedAndConvertToI8MixedPrecisionExplicitDequant
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x1x1x256xf16>
func.func @SkipFullQuantizedAndConvertToI8MixedPrecisionExplicitDequant(%arg0: tensor<1x1x1x256xf16>) -> (tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>) {
  %cst = const.Declare tensor<4x256x1x1x!qElemType> = dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x256x1x1x!qElemType1> = dense<1> : tensor<1x256x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
  %in_dequantize = IE.Dequantize(%cst_0) {dstElemType = f16} : tensor<1x256x1x1x!qElemType1> -> tensor<1x256x1x1xf16>
  %wt_dequantize = IE.Dequantize(%cst) {dstElemType = f16} : tensor<4x256x1x1x!qElemType> -> tensor<4x256x1x1xf16>
  %conv = IE.Convolution(%in_dequantize, %wt_dequantize) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1xf16> -> tensor<1x4x1x1xf16>
  %af_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
  %wt_dequantize_0 = IE.Dequantize(%cst) {dstElemType = f16} : tensor<4x256x1x1x!qElemType> -> tensor<4x256x1x1xf16>
  %conv_0 = IE.Convolution(%af_reshape, %wt_dequantize_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1xf16> -> tensor<1x4x1x1xf16>

  return %conv, %conv_0 : tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x256x1x1x!qElemType> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<4x256x1x1x!qElemType1> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.ConvertElemType<!qElemType1>]
  // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x256x1x1x!qElemType2> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<1x256x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>]
  // CHECK:  [[IN_DQ:%.+]] = IE.Dequantize([[CST_1]]) {dstElemType = f16} : tensor<1x256x1x1x!qElemType2> -> tensor<1x256x1x1xf16>
  // CHECK:  [[WT_DQ:%.+]] = IE.Dequantize([[CST_0]]) {dstElemType = f16} : tensor<4x256x1x1x!qElemType1> -> tensor<4x256x1x1xf16>
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[IN_DQ]], [[WT_DQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1xf16> -> tensor<1x4x1x1xf16>
  // CHECK:  [[AF_RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
  // CHECK:  [[WT_DQ_0:%.+]] = IE.Dequantize([[CST]]) {dstElemType = f16} : tensor<4x256x1x1x!qElemType> -> tensor<4x256x1x1xf16>
  // CHECK:  [[CONV_0:%.+]] = IE.Convolution([[AF_RESHAPE]], [[WT_DQ_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1xf16> -> tensor<1x4x1x1xf16>

  // CHECK:  return [[CONV]], [[CONV_0]] : tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>
}

// -----

// CHECK:     !qElemType = !quant.uniform<i8:f16:0, {5.4931640625E-4,6.7042553541707059E-4,5.74307172906165E-4,6.6369725208656461E-4}>
// CHECK:     !qElemType1 = !quant.uniform<u8:f16:0, {5.4931640625E-4:128,6.7042553541707059E-4:128,5.74307172906165E-4:128,6.6369725208656461E-4:128}>
// CHECK:     !qElemType2 = !quant.uniform<u8:f16, 3.1372548227070592E-7>

!qElemType = !quant.uniform<i8:f16:0, {5.4931640625E-4,6.7042553541707059E-4,5.74307172906165E-4,6.6369725208656461E-4}>
!qElemType1 = !quant.uniform<u8:f16, 3.1372548227070592E-7>

// CHECK-LABEL: @SkipFullQuantizedAndConvertToI8MixedPrecisionFusedDequantize
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x1x1x256xf16>
func.func @SkipFullQuantizedAndConvertToI8MixedPrecisionFusedDequantize(%arg0: tensor<1x1x1x256xf16>) -> (tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>) {
  %cst = const.Declare tensor<4x256x1x1x!qElemType> = dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
  %cst_0 = const.Declare tensor<1x256x1x1x!qElemType1> = dense<1> : tensor<1x256x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType1>]
  %conv = IE.Convolution(%cst_0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1x!qElemType1>, tensor<4x256x1x1x!qElemType> -> tensor<1x4x1x1xf16>
  %af_reshape = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
  %conv_0 = IE.Convolution(%af_reshape, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1x!qElemType> -> tensor<1x4x1x1xf16>

  return %conv, %conv_0 : tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>

  // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<4x256x1x1x!qElemType> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<4x256x1x1x!qElemType1> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<4x256x1x1xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.ConvertElemType<!qElemType1>]
  // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x256x1x1x!qElemType2> =
  // CHECK-SAME{LITERAL}:   dense<1> : tensor<1x256x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>]
  // CHECK:  [[CONV:%.+]] = IE.Convolution([[CST_1]], [[CST_0]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1x!qElemType2>, tensor<4x256x1x1x!qElemType1> -> tensor<1x4x1x1xf16>
  // CHECK:  [[AF_RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
  // CHECK:  [[CONV_0:%.+]] = IE.Convolution([[AF_RESHAPE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<4x256x1x1x!qElemType> -> tensor<1x4x1x1xf16>

  // CHECK:  return [[CONV]], [[CONV_0]] : tensor<1x4x1x1xf16>, tensor<1x4x1x1xf16>
}

// -----

// CHECK:     !qElemType = !quant.uniform<u8:f16, 5.000000e-01>
// CHECK:     !qElemType1 = !quant.uniform<u8:f16, 1.800000e-02:141>

!qElemType_u8 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType_u8_b = !quant.uniform<u8:f16, 1.800000e-02:141>
!qElemType_act = !quant.uniform<u8:f16, 0.500000e+00>

// CHECK-LABEL: @ChainedQuantizeCastU8ToQuantileU8
// CHECK-SAME:      [[ARG0:%.+]]: tensor<256x768x1x1xui8>
// CHECK-SAME:      [[ARG1:%.+]]: tensor<1x768x1x1x!qElemType>
func.func @ChainedQuantizeCastU8ToQuantileU8(%arg0: tensor<256x768x1x1xui8>, %arg1: tensor<1x768x1x1x!qElemType_act>) -> tensor<1x256x1x1xf16> {
  %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType_u8} : tensor<256x768x1x1xui8> -> tensor<256x768x1x1x!qElemType_u8>
  %1 = IE.QuantizeCast(%0) {dstElemType = !qElemType_u8_b} : tensor<256x768x1x1x!qElemType_u8> -> tensor<256x768x1x1x!qElemType_u8_b>
  %2 = IE.Dequantize(%1) {dstElemType = f16} : tensor<256x768x1x1x!qElemType_u8_b> -> tensor<256x768x1x1xf16>
  %3 = IE.Dequantize(%arg1) {dstElemType = f16} : tensor<1x768x1x1x!qElemType_act> -> tensor<1x768x1x1xf16>
  %4 = IE.Convolution(%3, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x768x1x1xf16>, tensor<256x768x1x1xf16> -> tensor<1x256x1x1xf16>
  return %4 : tensor<1x256x1x1xf16>

  // CHECK-NOT:   unrealized_conversion_cast

  // CHECK:       [[QUANT:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = !qElemType1}
  // CHECK-SAME:      : tensor<256x768x1x1xui8> -> tensor<256x768x1x1x!qElemType1>
  // CHECK:       [[DEQUANT_WTS:%.+]] = IE.Dequantize([[QUANT]]) {dstElemType = f16}
  // CHECK-SAME:      : tensor<256x768x1x1x!qElemType1> -> tensor<256x768x1x1xf16>
  // CHECK:       [[DEQUANT_ACT:%.+]] = IE.Dequantize([[ARG1]]) {dstElemType = f16}
  // CHECK-SAME:      : tensor<1x768x1x1x!qElemType> -> tensor<1x768x1x1xf16>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[DEQUANT_ACT]], [[DEQUANT_WTS]])
  // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:      : tensor<1x768x1x1xf16>, tensor<256x768x1x1xf16> -> tensor<1x256x1x1xf16>
  // CHECK:       return [[CONV]] : tensor<1x256x1x1xf16>
}

// -----

// Gather op with quantized I8 embedding table: the pass must NOT convert the
// constant or the Gather op to U8

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @KeepI8ConstantsForGather
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x512xsi32>
func.func @KeepI8ConstantsForGather(%arg0: tensor<1x512xsi32>) -> tensor<1x512x768x!qElemType> {
  %cst = const.Declare tensor<250002x768x!qElemType> = dense<0> : tensor<250002x768xsi8>,
      [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %0 = IE.Gather(%cst, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64}
      : tensor<250002x768x!qElemType>, tensor<1x512xsi32> -> tensor<1x512x768x!qElemType>
  return %0 : tensor<1x512x768x!qElemType>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<250002x768x!qElemType>
  // CHECK-NOT:   #const.ConvertElemType
  // CHECK:       [[GATHER:%.+]] = IE.Gather([[CST]], [[ARG0]])
  // CHECK-SAME:      {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64}
  // CHECK-SAME:      : tensor<250002x768x!qElemType>, tensor<1x512xsi32> -> tensor<1x512x768x!qElemType>
  // CHECK:       return [[GATHER]] : tensor<1x512x768x!qElemType>
}

// -----

// CHECK-DAG: !qElemType = !quant.uniform<i8:f16, 0.047244105488061905>
!qElemType = !quant.uniform<i8:f16, 0.047244105488061905>

// CHECK-LABEL: @KeepI8ConcatInputWithSI8BlockArg
// CHECK-SAME:      [[INPUT_0:%[^:]+]]: tensor<1x256x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%[^:]+]]: tensor<1x256x1x1xf16>
// CHECK-SAME:      [[SI8_ARG:%[^:]+]]: tensor<1x512x1x1xsi8>
func.func @KeepI8ConcatInputWithSI8BlockArg(%arg0: tensor<1x256x1x1xf16>, %arg1: tensor<1x256x1x1xf16>, %arg2: tensor<1x512x1x1xsi8>) -> tensor<1x768x1x1xf16> {
  %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  %qcast0 = IE.QuantizeCast(%add) {dstElemType = si8} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xsi8>
  %concat = IE.Concat(%qcast0, %arg2) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1xsi8>, tensor<1x512x1x1xsi8> -> tensor<1x768x1x1xsi8>
  %qcast1 = IE.QuantizeCast(%concat) {dstElemType = !qElemType} : tensor<1x768x1x1xsi8> -> tensor<1x768x1x1x!qElemType>
  %dequant = IE.Dequantize(%qcast1) {dstElemType = f16} : tensor<1x768x1x1x!qElemType> -> tensor<1x768x1x1xf16>
  return %dequant : tensor<1x768x1x1xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  // CHECK:       [[QCAST0:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = si8} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xsi8>
  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[QCAST0]], [[SI8_ARG]])
  // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0]]} : tensor<1x256x1x1xsi8>, tensor<1x512x1x1xsi8> -> tensor<1x768x1x1xsi8>
  // CHECK:       [[QCAST1:%.+]] = IE.QuantizeCast([[CONCAT]]) {dstElemType = !qElemType} : tensor<1x768x1x1xsi8> -> tensor<1x768x1x1x!qElemType>
  // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[QCAST1]]) {dstElemType = f16} : tensor<1x768x1x1x!qElemType> -> tensor<1x768x1x1xf16>
  // CHECK:       return [[DEQUANT]] : tensor<1x768x1x1xf16>
}

// -----

// Gather with a quantized I8 embedding table flowing through AffineReshape into
// DynamicDequantize: the entire chain must remain in I8 storage.
// DynamicDequantize consumes signed quantized input natively (its SHAVE kernel
// does not support U8)

!qElemType = !quant.uniform<i8:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-LABEL: @KeepI8GatherWithDynamicDequantize
// CHECK-SAME:      [[ARG0:%.+]]: tensor<1x512xsi32>
// CHECK-SAME:      [[SCALE:%.+]]: tensor<1x512x1x1xf16>
func.func @KeepI8GatherWithDynamicDequantize(
        %arg0: tensor<1x512xsi32>,
        %arg1: tensor<1x512x1x1xf16>) -> tensor<1x512x1x768xf16> {
  %cst = const.Declare tensor<250002x768x!qElemType> = dense<0> : tensor<250002x768xsi8>,
      [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  %0 = IE.Gather(%cst, %arg0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64}
      : tensor<250002x768x!qElemType>, tensor<1x512xsi32> -> tensor<1x512x768x!qElemType>
  %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 512, 1, 768]}
      : tensor<1x512x768x!qElemType> -> tensor<1x512x1x768x!qElemType>
  %2 = IE.DynamicDequantize(%1, %arg1) {dstElemType = f16}
      : tensor<1x512x1x768x!qElemType>, tensor<1x512x1x1xf16> -> tensor<1x512x1x768xf16>
  return %2 : tensor<1x512x1x768xf16>

  // CHECK:       [[CST:%.+]] = const.Declare tensor<250002x768x!qElemType>
  // CHECK-NOT:   #const.ConvertElemType
  // CHECK:       [[GATHER:%.+]] = IE.Gather([[CST]], [[ARG0]])
  // CHECK-SAME:      {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64}
  // CHECK-SAME:      : tensor<250002x768x!qElemType>, tensor<1x512xsi32> -> tensor<1x512x768x!qElemType>
  // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[GATHER]])
  // CHECK-SAME:      : tensor<1x512x768x!qElemType> -> tensor<1x512x1x768x!qElemType>
  // CHECK:       [[DEQUANT:%.+]] = IE.DynamicDequantize([[RESHAPE]], [[SCALE]]) {dstElemType = f16}
  // CHECK-SAME:      : tensor<1x512x1x768x!qElemType>, tensor<1x512x1x1xf16> -> tensor<1x512x1x768xf16>
  // CHECK:       return [[DEQUANT]] : tensor<1x512x1x768xf16>
}

// -----

// CHECK-DAG: !qElemType = !quant.uniform<i8:f16, 0.047244105488061905>
!qElemType = !quant.uniform<i8:f16, 0.047244105488061905>

// CHECK-LABEL: @KeepI8ConcatInputWithSI8BlockArgViaReshape
// CHECK-SAME:      [[INPUT_0:%[^:]+]]: tensor<1x256x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%[^:]+]]: tensor<1x256x1x1xf16>
// CHECK-SAME:      [[SI8_ARG:%[^:]+]]: tensor<512x1x1x1xsi8>
func.func @KeepI8ConcatInputWithSI8BlockArgViaReshape(%arg0: tensor<1x256x1x1xf16>, %arg1: tensor<1x256x1x1xf16>, %arg2: tensor<512x1x1x1xsi8>) -> tensor<768x1x1x1xf16> {
  %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  %qcast0 = IE.QuantizeCast(%add) {dstElemType = si8} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xsi8>
  %reshape = IE.AffineReshape(%qcast0) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 1, 1, 1]} : tensor<1x256x1x1xsi8> -> tensor<256x1x1x1xsi8>
  %concat = IE.Concat(%reshape, %arg2) {static_offsets = [[0, 0, 0, 0], [256, 0, 0, 0]]} : tensor<256x1x1x1xsi8>, tensor<512x1x1x1xsi8> -> tensor<768x1x1x1xsi8>
  %qcast1 = IE.QuantizeCast(%concat) {dstElemType = !qElemType} : tensor<768x1x1x1xsi8> -> tensor<768x1x1x1x!qElemType>
  %dequant = IE.Dequantize(%qcast1) {dstElemType = f16} : tensor<768x1x1x1x!qElemType> -> tensor<768x1x1x1xf16>
  return %dequant : tensor<768x1x1x1xf16>

  // CHECK:       [[ADD:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x1x1xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x1x!qElemType>
  // CHECK:       [[QCAST0:%.+]] = IE.QuantizeCast([[ADD]]) {dstElemType = si8} : tensor<1x256x1x1x!qElemType> -> tensor<1x256x1x1xsi8>
  // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[QCAST0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [256, 1, 1, 1]} : tensor<1x256x1x1xsi8> -> tensor<256x1x1x1xsi8>
  // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE]], [[SI8_ARG]])
  // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [256, 0, 0, 0]]} : tensor<256x1x1x1xsi8>, tensor<512x1x1x1xsi8> -> tensor<768x1x1x1xsi8>
  // CHECK:       [[QCAST1:%.+]] = IE.QuantizeCast([[CONCAT]]) {dstElemType = !qElemType} : tensor<768x1x1x1xsi8> -> tensor<768x1x1x1x!qElemType>
  // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[QCAST1]]) {dstElemType = f16} : tensor<768x1x1x1x!qElemType> -> tensor<768x1x1x1xf16>
  // CHECK:       return [[DEQUANT]] : tensor<768x1x1x1xf16>
}

// -----
!qElemTypeI8Act = !quant.uniform<i8:f16, 0.060064338235294119>
!qElemTypeU2Wt = !quant.uniform<u2:f16, 1.000000e+00:2>
!qElemTypeU2LUTWt = !quant.uniform<!QuantileType.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 1.000000e+00>

// CHECK: !qElemType = !quant.uniform<i8:f16, 0.060064338235294119>
// CHECK: !qElemType1 = !quant.uniform<u2:f16, 1.000000e+00:2>
// CHECK: !qElemType2 = !quant.uniform<!QuantileType.quantile<ui2:si8, {-2.000000e+00,-1.000000e+00,0.000000e+00,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @KeepSignedI8ActWhenWeightsAreU2LUT
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x1536xf32>
// CHECK-SAME:      [[WEIGHTS:%.+]]: tensor<16x1536xui2>
func.func @KeepSignedI8ActWhenWeightsAreU2LUT(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<16x1536xui2>) -> tensor<1x16x1x1xf16> {
  %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1x1x1536xf32>
  %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x1x1x1536xf32> -> tensor<1x1x1x1536xf16>
  %2 = IE.Quantize(%1) {dstElemType = !qElemTypeI8Act} : tensor<1x1x1x1536xf16> -> tensor<1x1x1x1536x!qElemTypeI8Act>
  %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 1536]} : tensor<16x1536xui2> -> tensor<1x1x16x1536xui2>
  %4 = IE.QuantizeCast(%3) {dstElemType = !qElemTypeU2Wt} : tensor<1x1x16x1536xui2> -> tensor<1x1x16x1536x!qElemTypeU2Wt>
  %5 = IE.AffineReshape(%2) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 1536, 1, 1]} : tensor<1x1x1x1536x!qElemTypeI8Act> -> tensor<1x1536x1x1x!qElemTypeI8Act>
  %6 = IE.AffineReshape(%4) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 1536, 1, 1]} : tensor<1x1x16x1536x!qElemTypeU2Wt> -> tensor<16x1536x1x1x!qElemTypeU2Wt>
  %7 = IE.QuantizeCast(%6) {dstElemType = !qElemTypeU2LUTWt} : tensor<16x1536x1x1x!qElemTypeU2Wt> -> tensor<16x1536x1x1x!qElemTypeU2LUTWt>
  %8 = IE.Convolution(%5, %7) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1x!qElemTypeI8Act>, tensor<16x1536x1x1x!qElemTypeU2LUTWt> -> tensor<1x16x1x1xf16>
  return %8 : tensor<1x16x1x1xf16>

  // CHECK:       [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1x1x1536xf32>
  // CHECK:       [[CONVERT:%.+]] = IE.Convert([[RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x1536xf32> -> tensor<1x1x1x1536xf16>
  // CHECK:       [[QUANTIZE:%.+]] = IE.Quantize([[CONVERT]]) {dstElemType = !qElemType} : tensor<1x1x1x1536xf16> -> tensor<1x1x1x1536x!qElemType>
  // CHECK-NOT:   builtin.unrealized_conversion_cast
  // CHECK:       [[RESHAPE_1:%.+]] = IE.AffineReshape([[WEIGHTS]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 16, 1536]} : tensor<16x1536xui2> -> tensor<1x1x16x1536xui2>
  // CHECK:       [[QCAST_0:%.+]] = IE.QuantizeCast([[RESHAPE_1]]) {dstElemType = !qElemType1} : tensor<1x1x16x1536xui2> -> tensor<1x1x16x1536x!qElemType1>
  // CHECK:       [[RESHAPE_2:%.+]] = IE.AffineReshape([[QUANTIZE]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 1536, 1, 1]} : tensor<1x1x1x1536x!qElemType> -> tensor<1x1536x1x1x!qElemType>
  // CHECK:       [[RESHAPE_3:%.+]] = IE.AffineReshape([[QCAST_0]])
  // CHECK-SAME{LITERAL}:   {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16, 1536, 1, 1]} : tensor<1x1x16x1536x!qElemType1> -> tensor<16x1536x1x1x!qElemType1>
  // CHECK:       [[QCAST_1:%.+]] = IE.QuantizeCast([[RESHAPE_3]]) {dstElemType = !qElemType2} : tensor<16x1536x1x1x!qElemType1> -> tensor<16x1536x1x1x!qElemType2>
  // CHECK:       [[CONV:%.+]] = IE.Convolution([[RESHAPE_2]], [[QCAST_1]])
  // CHECK-SAME:            {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1536x1x1x!qElemType>, tensor<16x1536x1x1x!qElemType2> -> tensor<1x16x1x1xf16>
  // CHECK:       return [[CONV]] : tensor<1x16x1x1xf16>
}
