//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOCWithoutAnyWeightTable
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOCWithoutAnyWeightTable(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<9216x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [9216, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,

        strides = [1, 1]
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<9216x32x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[4608, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]]) rawFilterShape [4608, 32, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]]) rawFilterShape [4608, 32, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 4608, 0, 0]
    // CHECK-SAME:          -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverIHSplitWeightTableFormat
// CHECK-SAME:          [[INPUT:%[^:]+]]: tensor<1x32x8704x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverIHSplitWeightTableFormat(%arg0: tensor<1x32x8704x16x!qElemType, {order = #NHWC}>) -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<64x1x1x1xf32, {order = #NCHW}> = dense<1.0> : tensor<64x1x1x1xf32>
    %bias_table = const.Declare tensor<64x1x1x1xf32, {order = #NCHW}> = dense<2.0> : tensor<64x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,

        strides = [2, 2],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x32x8704x16x!qElemType, {order = #NHWC}>, tensor<64x32x3x3x!qElemType2, {order = #NHWC}>, tensor<64x1x1x1xf32, {order = #NCHW}>, tensor<64x1x1x1xf32, {order = #NCHW}> -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:        [[SCALE_TABLE:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NCHW}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf32>
    // CHECK:        [[BIAS_TABLE:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NCHW}> = dense<2.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf32>

    // CHECK:        [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 4352, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType, {order = #NHWC}> to tensor<1x32x4352x16x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [64, 32, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4351, 0] [1, 32, 4353, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType, {order = #NHWC}> to tensor<1x32x4353x16x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [64, 32, 3, 3] {
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 2176, 0]
    // CHECK-SAME:          -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC2ConvsSplitWeightTableFormat
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC2ConvsSplitWeightTableFormat(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>
  %scale_table = const.Declare tensor<512x1x1x1xf32> = dense<0.7> : tensor<512x1x1x1xf32>
  %bias_table = const.Declare tensor<512x1x1x1xf32> = dense<2.0> : tensor<512x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [512, 9728, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,

    strides = [1, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>, tensor<512x1x1x1xf32>, tensor<512x1x1x1xf32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>]
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>]

  // CHECK-DAG:      [[SCALE_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<1.000000e+00> : tensor<512x1x1x1xf32>
  // CHECK-DAG:      [[BIAS_TABLE0:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<2.000000e+00> : tensor<512x1x1x1xf32>
  // CHECK-DAG:      [[BIAS_TABLE1:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<0.000000e+00> : tensor<512x1x1x1xf32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER0]], [[SCALE_TABLE]], [[BIAS_TABLE0]]) rawFilterShape [512, 4864, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[SCALE_TABLE]], [[BIAS_TABLE1]]) rawFilterShape [512, 4864, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 0.69999998807907104 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverICandOCSplitWeightTableFormat
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverICandOCSplitWeightTableFormat(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x9216x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9216x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>
  %scale_table = const.Declare tensor<9216x1x1x1xf32> = dense<1.0> : tensor<9216x1x1x1xf32>
  %bias_table = const.Declare tensor<9216x1x1x1xf32> = dense<2.0> : tensor<9216x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [9216, 16640, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,

    strides = [1, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<9216x16640x1x1xf16, {order = #NHWC}>, tensor<9216x1x1x1xf32>, tensor<9216x1x1x1xf32> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 0, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[FILTER2:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[FILTER3:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[FILTER4:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [4608, 5536, 1, 1]>]
  // CHECK-DAG:   [[FILTER5:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 11104, 0, 0], [4608, 5536, 1, 1]>]

  // CHECK-DAG:   [[SCALE_TABLE0:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<1.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 1]>]
  // CHECK-DAG:   [[SCALE_TABLE1:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<1.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 1]>]

  // CHECK-DAG:   [[BIAS_TABLE0:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<2.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 1]>]
  // CHECK-DAG:   [[BIAS_TABLE1:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<2.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 1]>]
  // CHECK-DAG:   [[BIAS_TABLE2:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<0.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 1]>]
  // CHECK-DAG:   [[BIAS_TABLE3:%.+]] = const.Declare tensor<4608x1x1x1xf32> = dense<0.000000e+00> : tensor<9216x1x1x1xf32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 1]>]

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER0]], [[SCALE_TABLE0]], [[BIAS_TABLE0]]) rawFilterShape [4608, 5552, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER1]], [[SCALE_TABLE1]], [[BIAS_TABLE1]]) rawFilterShape [4608, 5552, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>


  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0]], [[CONV_OUT1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER2]], [[SCALE_TABLE0]], [[BIAS_TABLE2]]) rawFilterShape [4608, 5552, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER3]], [[SCALE_TABLE1]], [[BIAS_TABLE3]]) rawFilterShape [4608, 5552, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2]], [[CONV_OUT3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER4]], [[SCALE_TABLE0]], [[BIAS_TABLE2]]) rawFilterShape [4608, 5536, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:     -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER5]], [[SCALE_TABLE1]], [[BIAS_TABLE3]]) rawFilterShape [4608, 5536, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4:%.+]], [[CONV_OUT5:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE3]], [[INPUT_SLICE4]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE5]], [[INPUT_SLICE6]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT0]], [[ADD_OUT1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT3]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7]], [[INPUT_SLICE8]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,  ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT3]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9]], [[INPUT_SLICE10]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT4:%.+]] = VPU.Concat([[ADD_OUT2]], [[ADD_OUT3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       return [[CONCAT_OUT4]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecisionI4SplitWeightTableFormat
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x16416x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecisionI4SplitWeightTableFormat(%arg0: tensor<1x16416x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16416x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<0.007874015733307484> : tensor<16x1x1x1xf32>
  %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<2.0> : tensor<16x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [16, 16416, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,

    strides = [1, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x16416x1x1xf16, {order = #NHWC}>, tensor<16x16416x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 10944, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 5472, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]

  // CHECK-DAG:  [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.00787401571> : tensor<16x1x1x1xf32>
  // CHECK-DAG:  [[BIAS_TABLE0:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
  // CHECK-DAG:  [[BIAS_TABLE1:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[SCALE_TABLE]], [[BIAS_TABLE0]]) rawFilterShape [16, 5472, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[SCALE_TABLE]], [[BIAS_TABLE1]]) rawFilterShape [16, 5472, 1, 1] {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[SCALE_TABLE]], [[BIAS_TABLE1]]) rawFilterShape [16, 5472, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64,
  // CHECK-SAME:                    scale = 0.99999999814465923 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Verify that IC splitting correctly forwards the reduce_xy_max output (and axes_value) from the
// original NCEConvolution to the accumulating NCEEltwise, while intermediate convolutions carry
// no reduce outputs of their own.
//
// CHECK-LABEL:   @SplitNCEConvWithReduceXyMaxOverIC
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvWithReduceXyMaxOverIC(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>)
    -> (tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>) {
  %weights = const.Declare tensor<512x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>
  %scale_table = const.Declare tensor<512x1x1x1xf32> = dense<7.000000e-01> : tensor<512x1x1x1xf32>
  %bias_table = const.Declare tensor<512x1x1x1xf32> = dense<2.000000e+00> : tensor<512x1x1x1xf32>
  %conv, %reduce_max = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table)
      rawFilterShape [512, 9728, 1, 1] {
          axes_value = [1],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64,
                           prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
          resultSegmentSizes = array<i32: 1, 1, 0, 0>,
          strides = [1, 1],
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
      } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>,
          tensor<512x1x1x1xf32>, tensor<512x1x1x1xf32>
        -> tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>
  return %conv, %reduce_max : tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>]
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>]
  // CHECK-DAG:      [[SCALE_TABLE:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<1.000000e+00> : tensor<512x1x1x1xf32>
  // CHECK-DAG:      [[BIAS_TABLE0:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<2.000000e+00> : tensor<512x1x1x1xf32>
  // CHECK-DAG:      [[BIAS_TABLE1:%.+]] = const.Declare tensor<512x1x1x1xf32> = dense<0.000000e+00> : tensor<512x1x1x1xf32>

  // CHECK:      [[IN_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1]
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[IN_SLICE0]], [[FILTER0]], [[SCALE_TABLE]], [[BIAS_TABLE0]]) rawFilterShape [512, 4864, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[IN_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1]
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[IN_SLICE1]], [[FILTER1]], [[SCALE_TABLE]], [[BIAS_TABLE1]]) rawFilterShape [512, 4864, 1, 1] {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:   strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // Last accumulating eltwise carries the reduce_xy_max output and axes_value from the original conv.
  // CHECK:      [[ADD_OUT:%.+]], [[RED_MAX:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   axes_value = [1]
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 0.69999998807907104 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT]], [[RED_MAX]] : tensor<1x512x4x1xf16, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Per-axis (per-channel) quantized output type. Only the *kind* of this type
// (mlir::quant::UniformQuantizedPerAxisType) drives the trailing NCEDepthConvolution
// insertion branch in splitNCEConvolutionOverIC; the scale values themselves are not
// otherwise checked below.
!qElemType = !quant.uniform<u8:f16:1, {5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01,5.000000e-01}>

// Verify that when IC splitting requires a trailing NCEDepthConvolution to apply a
// per-channel output scale, the reduce_xy_max output and axes_value are forwarded to
// that NCEDepthConvolution (not to the last accumulating NCEEltwise).
//
// CHECK-LABEL:   @SplitNCEConvWithReduceXyMaxOverICAndDWConv
// CHECK-SAME:    [[INPUT:%[^:]+]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvWithReduceXyMaxOverICAndDWConv(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>)
    -> (tensor<1x16x4x1x!qElemType, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>) {
  %weights = const.Declare tensor<16x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x9728x1x1xf16, {order = #NHWC}>
  %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<7.000000e-01> : tensor<16x1x1x1xf32>
  %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
  %conv, %reduce_max = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table)
      rawFilterShape [16, 9728, 1, 1] {
          axes_value = [1],
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64,
                           prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
          resultSegmentSizes = array<i32: 1, 1, 0, 0>,
          strides = [1, 1],
          operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
      } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<16x9728x1x1xf16, {order = #NHWC}>,
          tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
        -> tensor<1x16x4x1x!qElemType, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>
  return %conv, %reduce_max : tensor<1x16x4x1x!qElemType, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>


  // CHECK:      [[IN_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1]
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution
  // CHECK-SAME:   rawFilterShape [16, 4864, 1, 1] {
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[IN_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1]
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution
  // CHECK-SAME:   rawFilterShape [16, 4864, 1, 1] {
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 0, 0>,
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // The intermediate accumulating eltwise must NOT carry the reduce output or axes_value:
  // per-channel output scale means a trailing NCEDepthConvolution is required.
  // CHECK:      [[ADD_OUT:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 0, 0, 0>
  // CHECK-SAME: } -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // The trailing NCEDepthConvolution carries the reduce_xy_max output and axes_value
  // forwarded from the original convolution.
  // CHECK:      [[DW_OUT:%.+]], [[RED_MAX:%.+]] = VPU.NCE.DepthConvolution([[ADD_OUT]]
  // CHECK-SAME:   rawFilterShape [16, 1, 1, 1] {
  // CHECK-SAME:   axes_value = [1]
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   resultSegmentSizes = array<i32: 1, 1, 0, 0>
  // CHECK-SAME: } -> tensor<1x16x4x1x!qElemType, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[DW_OUT]], [[RED_MAX]] : tensor<1x16x4x1x!qElemType, {order = #NHWC}>, tensor<1x1x4x1xf16, {order = #NHWC}>
}
