//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<9216x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<9216x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<9216x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [9216, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<9216x32x3x3x!qElemType2, {order = #NHWC}>, tensor<9216x1x1x4xsi32, {order = #NCHW}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[4608, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [4608, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [4608, 32, 3, 3],
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

// CHECK-LABEL:   @SplitQuantNCEConvOverIH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x8704x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverIH(%arg0: tensor<1x32x8704x16x!qElemType, {order = #NHWC}>) -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [64, 32, 3, 3],
        strides = [2, 2]
    } : tensor<1x32x8704x16x!qElemType, {order = #NHWC}>, tensor<64x32x3x3x!qElemType2, {order = #NHWC}>, tensor<64x1x1x4xsi32, {order = #NCHW}> -> tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x64x4352x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<64x1x1x4xsi32>

    // CHECK:        [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 4352, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType, {order = #NHWC}> to tensor<1x32x4352x16x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4351, 0] [1, 32, 4353, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType, {order = #NHWC}> to tensor<1x32x4353x16x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
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

// CHECK-LABEL:   @SplitNCEConvOverIC2Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC2Convs(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x9728x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1097072640> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>,
  // CHECK-DAG-SAME:          [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>]
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>,
  // CHECK-DAG-SAME:          [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>]

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[9728, 608, 1065353216, 0]]], [[[19456, 1216, 1065353216, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1097072640]]], [[[9728, 608, 1065353216, 1097072640]]], [[[19456, 1216, 1065353216, 1097072640]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1]
  // CHECK-SAME:     : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.425000e+01], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}


// -----

!qElemType = !quant.uniform<u8:f16, 0.0028915546688379028:131>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvWithDequantizeOverIC2Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvWithDequantizeOverIC2Convs(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x9728x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1097072640> : tensor<512x1x1x4xsi32>
  %dequantize = VPU.Dequantize(%weights) {dstElemType = f16} :  tensor<512x9728x1x1x!qElemType, {order = #NHWC}> -> tensor<512x9728x1x1xf16, {order = #NHWC}>
  %0 = VPU.NCE.Convolution(%arg0, %dequantize, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1037483647 : i64, clamp_high = 1037483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 3.000000e-01 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1x!qElemType, {order = #NHWC}>
  // CHECK-DAG-SAME:   dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1x!qElemType, {order = #NHWC}>
  // CHECK-DAG-SAME:   dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[9728, 608, 1065353216, 0]]], [[[19456, 1216, 1065353216, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1097072640]]], [[[9728, 608, 1065353216, 1097072640]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1]
  // CHECK-SAME:     : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[DEQUANT0:%.+]] = VPU.Dequantize([[FILTER1]]) {dstElemType = f16}
  // CHECK-SAME:     : tensor<512x4864x1x1x!qElemType, {order = #NHWC}> -> tensor<512x4864x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[DEQUANT0]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1]
  // CHECK-SAME:     : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[DEQUANT1:%.+]] = VPU.Dequantize([[FILTER0]]) {dstElemType = f16}
  // CHECK-SAME:     : tensor<512x4864x1x1x!qElemType, {order = #NHWC}> -> tensor<512x4864x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[DEQUANT1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1037483647 : i64, clamp_high = 1037483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.425000e+01], fp_prelu_alpha = 3.000000e-01 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}


// -----

// Checking tiling retry logic, will generate 756 tiles. For slice and depthconv, check the first two and last two, ignor others.
// For concat, only check the first and last input, ignor others
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @CheckTilingRetryLogic
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x6193152x1x1xf16, {order = #NHWC}>
func.func @CheckTilingRetryLogic(%arg0: tensor<1x6193152x1x1xf16, {order = #NHWC}>,
                                %arg1: tensor<6193152x16x1x1xf16, {order = #NHWC}>,
                                %arg2: tensor<6193152x1x1x4xsi32, {order = #NCHW}>) -> tensor<1x6193152x1x1xf16, {order = #NHWC}> {
  %0 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [6193152, 1, 1, 1],
    strides = [1, 1]} -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_FIRST:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_FIRST:%.+]] = VPU.Slice %arg1 [0, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_FIRST:%.+]] = VPU.Slice %arg2 [0, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_FIRST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_FIRST]], [[WEIGHTS_SLICE_FIRST]], [[WEIGHTSTABLE_SLICE_FIRST]])
   //CHECK-SAME:    {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
   //CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
   //CHECK-SAME:     rawFilterShape = [8192, 1, 1, 1], strides = [1, 1]} -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_1:%.+]]  = VPU.Slice %arg0 [0, 8192, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_1:%.+]] = VPU.Slice %arg1 [8192, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_1:%.+]] = VPU.Slice %arg2 [8192, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_1:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_1]], [[WEIGHTS_SLICE_1]], [[WEIGHTSTABLE_SLICE_1]])
   //CHECK-SAME:    {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
   //CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
   //CHECK-SAME:     rawFilterShape = [8192, 1, 1, 1], strides = [1, 1]} -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_754:%.+]] = VPU.Slice %arg0 [0, 6176768, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_754:%.+]] = VPU.Slice %arg1 [6176768, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_754:%.+]] = VPU.Slice %arg2 [6176768, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_754:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_754]], [[WEIGHTS_SLICE_754]], [[WEIGHTSTABLE_SLICE_754]])
   //CHECK-SAME:    {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
   //CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
   //CHECK-SAME:     rawFilterShape = [8192, 1, 1, 1], strides = [1, 1]} -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_LAST:%.+]] = VPU.Slice %arg0 [0, 6184960, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_LAST:%.+]] = VPU.Slice %arg1 [6184960, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_LAST:%.+]] = VPU.Slice %arg2 [6184960, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_LAST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_LAST]], [[WEIGHTS_SLICE_LAST]], [[WEIGHTSTABLE_SLICE_LAST]])
   //CHECK-SAME:    {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
   //CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
   //CHECK-SAME:     rawFilterShape = [8192, 1, 1, 1], strides = [1, 1]} -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV_FIRST]],
   //CHECK-NOT:      DEPTHCONV_1
   //CHECK-NOT:      DEPTHCONV_754
   //CHECK-SAME:     [[DEPTHCONV_LAST]])
   //CHECK-SAME:     -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    return  [[CONCAT:%.+]] tensor<1x6193152x1x1xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEAveragePoolOverOW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x3x8832xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverOW(%arg0: tensor<1x16x3x8832xf16, {order = #NHWC}>) -> tensor<1x16x1x8832xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [3, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>,
               clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64,
               fp_prelu_alpha = 1.000000e+00 : f64,
               lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
               quant_scale = [0.33333333333333331]>,
        strides = [1, 1]
        } -> tensor<1x16x1x8832xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_0]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.33333333333333331], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:     strides = [1, 1]} -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4416] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_1]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.33333333333333331], fp_prelu_alpha = 1.000000e+00 : f64>,
    // CHECK-SAME:     strides = [1, 1]} -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 0, 4416]
    // CHECK-SAME:          -> tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8832xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC3Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3Convs(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<512x16640x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1]]], [[[11072, 704, 1065353216, 1]]], [[[22144, 1408, 1065353216, 1]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.4012984643248171E-44], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsWithOutNCHW
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsWithOutNCHW(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NCHW}> {
  %weights = const.Declare tensor<512x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<512x16640x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NCHW}>

  return %0 : tensor<1x512x4x1xf16, {order = #NCHW}>


  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1]]], [[[11072, 704, 1065353216, 1]]], [[[22144, 1408, 1065353216, 1]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [512, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                      fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     quant_scale = [1.4012984643248171E-44], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NCHW}>

  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverICandOC
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverICandOC(%arg0: tensor<1x16640x4x1xf16, {order = #NHWC}>) -> tensor<1x9216x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<9216x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>
  %weights_table = const.Declare tensor<9216x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 2.000000e-01 : f64>,
    rawFilterShape = [9216, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<9216x16640x1x1xf16, {order = #NHWC}>, tensor<9216x1x1x4xsi32> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER0:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 11104, 0, 0], [4608, 5536, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER1:%.+]] = const.Declare tensor<4608x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [4608, 5536, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER2:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE3:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER3:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE4:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER4:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[4608, 0, 0, 0], [4608, 5552, 1, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TABLE5:%.+]] = const.Declare tensor<4608x1x1x4xsi32>
  // CHECK-DAG:   [[FILTER5:%.+]] = const.Declare tensor<4608x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [4608, 5552, 1, 1]>]

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER5:%.+]], [[WEIGHTS_TABLE5:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER4:%.+]], [[WEIGHTS_TABLE4:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>


  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice %arg0 [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER3:%.+]], [[WEIGHTS_TABLE3:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2:%.+]], [[CONV_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice %arg0 [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:    rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4:%.+]], [[CONV_OUT5:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE3:%.+]], [[INPUT_SLICE4:%.+]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE5:%.+]], [[INPUT_SLICE6:%.+]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT0:%.+]], [[ADD_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7:%.+]], [[INPUT_SLICE8:%.+]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.4012984643248171E-44], fp_prelu_alpha = 2.000000e-01 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9:%.+]], [[INPUT_SLICE10:%.+]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.4012984643248171E-44], fp_prelu_alpha = 2.000000e-01 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT4:%.+]] = VPU.Concat([[ADD_OUT2:%.+]], [[ADD_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       return [[CONCAT_OUT4:%.+]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecision
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecision(%arg0: tensor<1x16640x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16640x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // scale value = 1038470039 = 0x3de5cb97 = 0.11220472419963165 = 14.25f (static_scale) * 0.007874015733307484 (weights_scale)
  %weights_table = const.Declare tensor<16x1x1x4xsi32> =
      dense<[[[[0, 0, 1038470039, 1]]], [[[16640, 2080, 1038470039, 2]]], [[[33280, 4160, 1038470039, 3]]],
            [[[49920, 6240, 1038470039, 4]]], [[[66560, 8320, 1038470039, 5]]], [[[83200, 10400, 1038470039, 6]]],
            [[[99840, 12480, 1038470039, 7]]], [[[116480, 14560, 1038470039, 8]]], [[[133120, 16640, 1038470039, 9]]],
            [[[149760, 18720, 1038470039, 10]]], [[[166400, 20800, 1038470039, 11]]], [[[183040, 22880, 1038470039, 12]]],
            [[[199680, 24960, 1038470039, 13]]], [[[216320, 27040, 1038470039, 14]]], [[[232960, 29120, 1038470039, 15]]],
            [[[249600, 31200, 1038470039, 16]]]]> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.0007874015733307484 : f64>,
    rawFilterShape = [16, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x1x1xf16, {order = #NHWC}>, tensor<16x16640x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5536x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [16, 5536, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [16, 5552, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5552, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]

  // Scale in weights table is now equal to weights scale 0.0007874015733307484 (= 0x3c010204 = 1006699012)
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1006699012, 0]]], [[[5536, 704, 1006699012, 0]]], [[[11072, 1408, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[16608, 2112, 1006699012, 0]]], [[[22144, 2816, 1006699012, 0]]], [[[27680, 3520, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[33216, 4224, 1006699012, 0]]], [[[38752, 4928, 1006699012, 0]]], [[[44288, 5632, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[49824, 6336, 1006699012, 0]]], [[[55360, 7040, 1006699012, 0]]], [[[60896, 7744, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[66432, 8448, 1006699012, 0]]], [[[71968, 9152, 1006699012, 0]]], [[[77504, 9856, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[83040, 10560, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>

  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1006699012, 0]]], [[[5552, 704, 1006699012, 0]]], [[[11104, 1408, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[16656, 2112, 1006699012, 0]]], [[[22208, 2816, 1006699012, 0]]], [[[27760, 3520, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[33312, 4224, 1006699012, 0]]], [[[38864, 4928, 1006699012, 0]]], [[[44416, 5632, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[49968, 6336, 1006699012, 0]]], [[[55520, 7040, 1006699012, 0]]], [[[61072, 7744, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[66624, 8448, 1006699012, 0]]], [[[72176, 9152, 1006699012, 0]]], [[[77728, 9856, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[83280, 10560, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>

  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1006699012, 1]]], [[[5552, 704, 1006699012, 2]]], [[[11104, 1408, 1006699012, 3]]],
  // CHECK-SAME{LITERAL}:             [[[16656, 2112, 1006699012, 4]]], [[[22208, 2816, 1006699012, 5]]], [[[27760, 3520, 1006699012, 6]]],
  // CHECK-SAME{LITERAL}:             [[[33312, 4224, 1006699012, 7]]], [[[38864, 4928, 1006699012, 8]]], [[[44416, 5632, 1006699012, 9]]],
  // CHECK-SAME{LITERAL}:             [[[49968, 6336, 1006699012, 10]]], [[[55520, 7040, 1006699012, 11]]], [[[61072, 7744, 1006699012, 12]]],
  // CHECK-SAME{LITERAL}:             [[[66624, 8448, 1006699012, 13]]], [[[72176, 9152, 1006699012, 14]]], [[[77728, 9856, 1006699012, 15]]],
  // CHECK-SAME{LITERAL}:             [[[83280, 10560, 1006699012, 16]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5536x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [14.249999855283427],
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e-01 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:       return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecisionI4
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16416x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecisionI4(%arg0: tensor<1x16416x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16416x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // scale value = 1038470039 = 0x3de5cb97 = 0.11220472419963165 = 14.25f (static_scale) * 0.007874015733307484 (weights_scale)
  %weights_table = const.Declare tensor<16x1x1x4xsi32> =
      dense<[[[[0, 0, 1038470039, 1]]], [[[16640, 2080, 1038470039, 2]]], [[[33280, 4160, 1038470039, 3]]],
            [[[49920, 6240, 1038470039, 4]]], [[[66560, 8320, 1038470039, 5]]], [[[83200, 10400, 1038470039, 6]]],
            [[[99840, 12480, 1038470039, 7]]], [[[116480, 14560, 1038470039, 8]]], [[[133120, 16640, 1038470039, 9]]],
            [[[149760, 18720, 1038470039, 10]]], [[[166400, 20800, 1038470039, 11]]], [[[183040, 22880, 1038470039, 12]]],
            [[[199680, 24960, 1038470039, 13]]], [[[216320, 27040, 1038470039, 14]]], [[[232960, 29120, 1038470039, 15]]],
            [[[249600, 31200, 1038470039, 16]]]]> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.0007874015733307484 : f64>,
    rawFilterShape = [16, 16416, 1, 1],
    strides = [1, 1]
  } : tensor<1x16416x1x1xf16, {order = #NHWC}>, tensor<16x16416x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 10944, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 5472, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]

  // Scale in weights table is now equal to weights scale 0.0007874015733307484 (= 0x3c010204 = 1006699012)
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1006699012, 0]]], [[[2736, 352, 1006699012, 0]]], [[[5472, 704, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[8208, 1056, 1006699012, 0]]], [[[10944, 1408, 1006699012, 0]]], [[[13680, 1760, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[16416, 2112, 1006699012, 0]]], [[[19152, 2464, 1006699012, 0]]], [[[21888, 2816, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[24624, 3168, 1006699012, 0]]], [[[27360, 3520, 1006699012, 0]]], [[[30096, 3872, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[32832, 4224, 1006699012, 0]]], [[[35568, 4576, 1006699012, 0]]], [[[38304, 4928, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:             [[[41040, 5280, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>

  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:       dense<[[[[0, 0, 1006699012, 1]]], [[[2736, 352, 1006699012, 2]]], [[[5472, 704, 1006699012, 3]]],
  // CHECK-SAME{LITERAL}:             [[[8208, 1056, 1006699012, 4]]], [[[10944, 1408, 1006699012, 5]]], [[[13680, 1760, 1006699012, 6]]],
  // CHECK-SAME{LITERAL}:             [[[16416, 2112, 1006699012, 7]]], [[[19152, 2464, 1006699012, 8]]], [[[21888, 2816, 1006699012, 9]]],
  // CHECK-SAME{LITERAL}:             [[[24624, 3168, 1006699012, 10]]], [[[27360, 3520, 1006699012, 11]]], [[[30096, 3872, 1006699012, 12]]],
  // CHECK-SAME{LITERAL}:             [[[32832, 4224, 1006699012, 13]]], [[[35568, 4576, 1006699012, 14]]], [[[38304, 4928, 1006699012, 15]]],
  // CHECK-SAME{LITERAL}:             [[[41040, 5280, 1006699012, 16]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -1020231680 : i64, clamp_high = 1128988672 : i64,
  // CHECK-SAME:                     lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
  // CHECK-SAME:                     quant_scale = [14.249999855283427], fp_prelu_alpha = 1.000000e-01 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvWithPerTensorQuantScale
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16384x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvWithPerTensorQuantScale(%arg0: tensor<1x16384x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<4096x16384x1x1xf16, {order = #NHWC}> = dense<64.0> : tensor<4096x16384x1x1xf16, {order = #NHWC}>
  // scale value = 1048403968 = 0x3e7d6000 = 0.2474365234375
  %weights_table = const.Declare tensor<4096x1x1x4xsi32> = dense<1048403968> : tensor<4096x1x1x4xsi32>

  %conv = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.2474365234375],
                      fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [4096, 16384, 1, 1], strides = [1, 1]
    } : tensor<1x16384x1x1xf16, {order = #NHWC}>, tensor<4096x16384x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32>
    -> tensor<1x4096x1x1xf16, {order = #NHWC}>

  return %conv : tensor<1x4096x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<4096x8192x1x1xf16, {order = #NHWC}>
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<4096x8192x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<4096x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 1065353216]]], [[[11072, 704, 1065353216, 1065353216]]], [[[22144, 1408, 1065353216, 1065353216]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<4096x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8192, 1, 1] : tensor<1x16384x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:   -> tensor<1x4096x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 8192, 0, 0] [1, 8192, 1, 1] : tensor<1x16384x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME:   -> tensor<1x4096x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [0.2474365234375], fp_prelu_alpha = 1.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x4096x1x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x4096x1x1xf16, {order = #NHWC}>
}
