//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
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
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [4608, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
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
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    // CHECK-SAME:          rawFilterShape = [64, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x64x2176x8x!qElemType1, {order = #NHWC}>

    // CHECK:        [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 4351, 0] [1, 32, 4353, 16]
    // CHECK-SAME:      : tensor<1x32x8704x16x!qElemType, {order = #NHWC}> to tensor<1x32x4353x16x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
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
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -2.4028234663852886E+10 : f64, clamp_high = 1.4028234663852886E+15 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>]
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>]

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[9728, 608, 1065353216, 0]]], [[[19456, 1216, 1065353216, 0]]], [[[29184, 1824, 1065353216, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 10]]], [[[9728, 608, 1065353216, 10]]], [[[19456, 1216, 1065353216, 10]]], [[[29184, 1824, 1065353216, 10]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER0]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -24028234663.852886 : f64, clamp_high = 1402823466385288.5 : f64,
  // CHECK-SAME:                     scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEAveragePoolOverOW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x3x8832xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverOW(%arg0: tensor<1x16x3x8832xf16, {order = #NHWC}>) -> tensor<1x16x1x8832xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [3, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
               clamp_high = 3.4028234663852886E+38 : f64, clamp_low = -3.4028234663852886E+38 : f64,
               scale = 1.000000e+00 : f64,
               prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64,
               adder = 0.000000e+00 : f64>,
        strides = [1, 1]
        } -> tensor<1x16x1x8832xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x8832xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_0]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:     strides = [1, 1]} -> tensor<1x16x1x4416xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4416] [1, 16, 3, 4416]
    // CHECK-SAME:      : tensor<1x16x3x8832xf16, {order = #NHWC}> to tensor<1x16x3x4416xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[ACTIVATION_TILE_1]]) {kernel_size = [3, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
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
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<512x16640x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11072, 704, 1065353216, 0]]], [[[22144, 1408, 1065353216, 0]]], [[[33216, 2112, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]], [[[33312, 2112, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 10]]], [[[11104, 704, 1065353216, 10]]], [[[22208, 1408, 1065353216, 10]]], [[[33312, 2112, 1065353216, 10]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
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
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<512x16640x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NCHW}>

  return %0 : tensor<1x512x4x1xf16, {order = #NCHW}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11072, 704, 1065353216, 0]]], [[[22144, 1408, 1065353216, 0]]], [[[33216, 2112, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 0]]], [[[11104, 704, 1065353216, 0]]], [[[22208, 1408, 1065353216, 0]]], [[[33312, 2112, 1065353216, 0]]]
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG-SAME{LITERAL}: dense<[[[[0, 0, 1065353216, 10]]], [[[11104, 704, 1065353216, 10]]], [[[22208, 1408, 1065353216, 10]]], [[[33312, 2112, 1065353216, 10]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NCHW}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NCHW}>
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
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
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

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER5]], [[WEIGHTS_TABLE5]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER4]], [[WEIGHTS_TABLE4]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>


  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0]], [[CONV_OUT1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER3]], [[WEIGHTS_TABLE3]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:    rawFilterShape = [4608, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2]], [[CONV_OUT3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:     rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:     -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64
  // CHECK-SAME:    rawFilterShape = [4608, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4]], [[CONV_OUT5]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE3]], [[INPUT_SLICE4]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE5]], [[INPUT_SLICE6]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT0]], [[ADD_OUT1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT3]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7]], [[INPUT_SLICE8]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT3]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9]], [[INPUT_SLICE10]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT4:%.+]] = VPU.Concat([[ADD_OUT2]], [[ADD_OUT3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       return [[CONCAT_OUT4]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.007874015733307484>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsMixedPrecision
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x1x1xf16, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsMixedPrecision(%arg0: tensor<1x16640x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<16x16640x1x1x!qElemType, {order = #NHWC}> = dense<64.0> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // scale = 1082549862 = 0x40866666 = 4.2f
  // bias will have the same value
  %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1082549862> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [0.0015748031466614968], adder = 0.000000e+00 : f64>,
    rawFilterShape = [16, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x1x1xf16, {order = #NHWC}>, tensor<16x16640x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5536x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [16, 5536, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [16, 5552, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5552x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16640x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5552, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType>]

  // Weights scale is found in weights table 1006699012 = 0x3c010204 = 0.007874016f
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1006699012, 0]]], [[[5536, 704, 1006699012, 0]]], [[[11072, 1408, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[16608, 2112, 1006699012, 0]]], [[[22144, 2816, 1006699012, 0]]], [[[27680, 3520, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[33216, 4224, 1006699012, 0]]], [[[38752, 4928, 1006699012, 0]]], [[[44288, 5632, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[49824, 6336, 1006699012, 0]]], [[[55360, 7040, 1006699012, 0]]], [[[60896, 7744, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[66432, 8448, 1006699012, 0]]], [[[71968, 9152, 1006699012, 0]]], [[[77504, 9856, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[83040, 10560, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1006699012, 0]]], [[[5552, 704, 1006699012, 0]]], [[[11104, 1408, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[16656, 2112, 1006699012, 0]]], [[[22208, 2816, 1006699012, 0]]], [[[27760, 3520, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[33312, 4224, 1006699012, 0]]], [[[38864, 4928, 1006699012, 0]]], [[[44416, 5632, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[49968, 6336, 1006699012, 0]]], [[[55520, 7040, 1006699012, 0]]], [[[61072, 7744, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[66624, 8448, 1006699012, 0]]], [[[72176, 9152, 1006699012, 0]]], [[[77728, 9856, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[83280, 10560, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1006699012, 1082549862]]], [[[5552, 704, 1006699012, 1082549862]]], [[[11104, 1408, 1006699012, 1082549862]]],
  // CHECK-SAME{LITERAL}:        [[[16656, 2112, 1006699012, 1082549862]]], [[[22208, 2816, 1006699012, 1082549862]]], [[[27760, 3520, 1006699012, 1082549862]]],
  // CHECK-SAME{LITERAL}:        [[[33312, 4224, 1006699012, 1082549862]]], [[[38864, 4928, 1006699012, 1082549862]]], [[[44416, 5632, 1006699012, 1082549862]]],
  // CHECK-SAME{LITERAL}:        [[[49968, 6336, 1006699012, 1082549862]]], [[[55520, 7040, 1006699012, 1082549862]]], [[[61072, 7744, 1006699012, 1082549862]]],
  // CHECK-SAME{LITERAL}:        [[[66624, 8448, 1006699012, 1082549862]]], [[[72176, 9152, 1006699012, 1082549862]]], [[[77728, 9856, 1006699012, 1082549862]]],
  // CHECK-SAME{LITERAL}:        [[[83280, 10560, 1006699012, 1082549862]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5552x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5552, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 1, 1] : tensor<1x16640x1x1xf16, {order = #NHWC}> to tensor<1x5536x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]]  = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5536, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:            prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}

  // Scale in orig op is 4.2f, weights scale is 0.007874015733307484
  // Therefore the static scale is 4.2/0.007874015733307484 = ~533.4f which is applied through this eltwise
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64,
  // CHECK-SAME:             scale = 533.39997677410338 : f64, prelu_alpha = [0.0015748031466614968], adder = 0.000000e+00 : f64>
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
  %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1065353216> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <TANH>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [16, 16416, 1, 1],
    strides = [1, 1]
  } : tensor<1x16416x1x1xf16, {order = #NHWC}>, tensor<16x16416x1x1x!qElemType, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 10944, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 5472, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5472x1x1x!qElemType, {order = #NHWC}> = dense<6.400000e+01> : tensor<16x16416x1x1xf32, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5472, 1, 1]>, #const.CastElemType<f16>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>]

  // Weights scale is found in weights table 1006699012 = 0x3c010204 = 0.007874016f
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1006699012, 0]]], [[[2736, 352, 1006699012, 0]]], [[[5472, 704, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[8208, 1056, 1006699012, 0]]], [[[10944, 1408, 1006699012, 0]]], [[[13680, 1760, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[16416, 2112, 1006699012, 0]]], [[[19152, 2464, 1006699012, 0]]], [[[21888, 2816, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[24624, 3168, 1006699012, 0]]], [[[27360, 3520, 1006699012, 0]]], [[[30096, 3872, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[32832, 4224, 1006699012, 0]]], [[[35568, 4576, 1006699012, 0]]], [[[38304, 4928, 1006699012, 0]]],
  // CHECK-SAME{LITERAL}:        [[[41040, 5280, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1006699012, 1065353216]]], [[[2736, 352, 1006699012, 1065353216]]], [[[5472, 704, 1006699012, 1065353216]]],
  // CHECK-SAME{LITERAL}:        [[[8208, 1056, 1006699012, 1065353216]]], [[[10944, 1408, 1006699012, 1065353216]]], [[[13680, 1760, 1006699012, 1065353216]]],
  // CHECK-SAME{LITERAL}:        [[[16416, 2112, 1006699012, 1065353216]]], [[[19152, 2464, 1006699012, 1065353216]]], [[[21888, 2816, 1006699012, 1065353216]]],
  // CHECK-SAME{LITERAL}:        [[[24624, 3168, 1006699012, 1065353216]]], [[[27360, 3520, 1006699012, 1065353216]]], [[[30096, 3872, 1006699012, 1065353216]]],
  // CHECK-SAME{LITERAL}:        [[[32832, 4224, 1006699012, 1065353216]]], [[[35568, 4576, 1006699012, 1065353216]]], [[[38304, 4928, 1006699012, 1065353216]]],
  // CHECK-SAME{LITERAL}:        [[[41040, 5280, 1006699012, 1065353216]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:    rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:    -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1] : tensor<1x16416x1x1xf16, {order = #NHWC}> to tensor<1x5472x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [16, 5472, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <TANH>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64, scale = 127.00000023748359 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x1x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x16x1x1xf16, {order = #NHWC}>
}

// -----
!qElemType = !quant.uniform<u8:f16, 0.0028915546688379028:131>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvWithDequantizeOverIC2Convs
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1xf16, {order = #NHWC}>
func.func @SplitNCEConvWithDequantizeOverIC2Convs(%arg0: tensor<1x9728x4x1xf16, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<512x9728x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<512x1x1x4xsi32>
  %dequantize = VPU.Dequantize(%weights) {dstElemType = f16} :  tensor<512x9728x1x1x!qElemType, {order = #NHWC}> -> tensor<512x9728x1x1xf16, {order = #NHWC}>
  %0 = VPU.NCE.Convolution(%arg0, %dequantize, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1xf16, {order = #NHWC}>, tensor<512x9728x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[FILTER0:%.+]] = const.Declare tensor<512x4864x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 4864, 0, 0], [512, 4864, 1, 1]>
  // CHECK-DAG:      [[FILTER1:%.+]] = const.Declare tensor<512x4864x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x9728x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 4864, 1, 1]>

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[DEQUANT0:%.+]] = VPU.Dequantize([[FILTER1]]) {dstElemType = f16} : tensor<512x4864x1x1x!qElemType, {order = #NHWC}> -> tensor<512x4864x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[DEQUANT0]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1xf16, {order = #NHWC}> to tensor<1x4864x4x1xf16, {order = #NHWC}>
  // CHECK:      [[DEQUANT1:%.+]] = VPU.Dequantize([[FILTER0]]) {dstElemType = f16} : tensor<512x4864x1x1x!qElemType, {order = #NHWC}> -> tensor<512x4864x1x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[DEQUANT1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]
  // CHECK-SAME: }
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:         scale = 1.4012984643248171E-44 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<f8E4M3FN:f16, 0.043808600732258389>

// CHECK-LABEL:   @SplitNCEConvOverIC3ConvsWithoutNCHWAndPerTensorScale
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x16640x4x1x!qElemType, {order = #NHWC}>
func.func @SplitNCEConvOverIC3ConvsWithoutNCHWAndPerTensorScale(%arg0: tensor<1x16640x4x1x!qElemType, {order = #NHWC}>) -> tensor<1x16x4x1xf16, {order = #NCHW}> {
  %weights = const.Declare tensor<16x16640x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16640x1x1xf16, {order = #NHWC}>
  // scale = 1054887119 = 10 (static scale) * 0.043808600732258389 (weights scale)
  %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1054887119> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <TANH>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64,
                     scale = 0.043808600732258389 : f64, bias = 2.34 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
    rawFilterShape = [16, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1x!qElemType, {order = #NHWC}>, tensor<16x16640x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x1xf16, {order = #NCHW}>

  return %0 : tensor<1x16x4x1xf16, {order = #NCHW}>


  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<16x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [16, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<16x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [16, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<16x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [16, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1026781350, 0]]], [[[11072, 704, 1026781350, 0]]], [[[22144, 1408, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[33216, 2112, 1026781350, 0]]], [[[44288, 2816, 1026781350, 0]]], [[[55360, 3520, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[66432, 4224, 1026781350, 0]]], [[[77504, 4928, 1026781350, 0]]], [[[88576, 5632, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[99648, 6336, 1026781350, 0]]], [[[110720, 7040, 1026781350, 0]]], [[[121792, 7744, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[132864, 8448, 1026781350, 0]]], [[[143936, 9152, 1026781350, 0]]], [[[155008, 9856, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[166080, 10560, 1026781350, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1026781350, 0]]], [[[11104, 704, 1026781350, 0]]], [[[22208, 1408, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[33312, 2112, 1026781350, 0]]], [[[44416, 2816, 1026781350, 0]]], [[[55520, 3520, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[66624, 4224, 1026781350, 0]]], [[[77728, 4928, 1026781350, 0]]], [[[88832, 5632, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[99936, 6336, 1026781350, 0]]], [[[111040, 7040, 1026781350, 0]]], [[[122144, 7744, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[133248, 8448, 1026781350, 0]]], [[[144352, 9152, 1026781350, 0]]], [[[155456, 9856, 1026781350, 0]]],
  // CHECK-SAME{LITERAL}:        [[[166560, 10560, 1026781350, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1026781350, 1054887119]]], [[[11104, 704, 1026781350, 1054887119]]], [[[22208, 1408, 1026781350, 1054887119]]],
  // CHECK-SAME{LITERAL}:        [[[33312, 2112, 1026781350, 1054887119]]], [[[44416, 2816, 1026781350, 1054887119]]], [[[55520, 3520, 1026781350, 1054887119]]],
  // CHECK-SAME{LITERAL}:        [[[66624, 4224, 1026781350, 1054887119]]], [[[77728, 4928, 1026781350, 1054887119]]], [[[88832, 5632, 1026781350, 1054887119]]],
  // CHECK-SAME{LITERAL}:        [[[99936, 6336, 1026781350, 1054887119]]], [[[111040, 7040, 1026781350, 1054887119]]], [[[122144, 7744, 1026781350, 1054887119]]],
  // CHECK-SAME{LITERAL}:        [[[133248, 8448, 1026781350, 1054887119]]], [[[144352, 9152, 1026781350, 1054887119]]], [[[155456, 9856, 1026781350, 1054887119]]],
  // CHECK-SAME{LITERAL}:        [[[166560, 10560, 1026781350, 1054887119]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1x!qElemType, {order = #NHWC}> to tensor<1x5552x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[FILTER2]], [[WEIGHTS_TABLE2]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.043808600732258389 : f64, prelu_alpha = [1.000000e+00], bias = 2.340000e+00 : f64, adder = 0.000000e+00 : f64>,

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1x!qElemType, {order = #NHWC}> to tensor<1x5552x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[FILTER1]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.043808600732258389 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1x!qElemType, {order = #NHWC}> to tensor<1x5536x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[FILTER0]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.043808600732258389 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:                    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64
  // CHECK-SAME: } -> tensor<1x16x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0]], [[CONV_OUT2]]) {
  // CHECK-SAME:   op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <TANH>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64,
  // CHECK-SAME:         scale = 9.9999999028164659 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x4x1xf16, {order = #NCHW}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x16x4x1xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.1>
!qElemType1 = !quant.uniform<i8:f32, 0.055118110030889511>

// CHECK-LABEL:   @SplitNCEConvOverIC2ConvsWithScaledWeights
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:    [[WEIGHTS:%arg[0-9]]]: tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>
func.func @SplitNCEConvOverIC2ConvsWithScaledWeights(%arg0: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, %arg1: tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x512x4x1xf16, {order = #NHWC}> {
  // scale = 1001692268 = 0x3bb49c6c = 0.005511811003088951f = 0.1 * 0.055118110030889511 / 1
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1001692268> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.25E+20 : f64, clamp_high = 4.4E+20 : f64, scale = 0.0055118110030889511 : f64,
                     prelu_alpha = [1.000000e-01], bias = 0.005511811003088951 : f64, adder = 0.000000e+00 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NHWC}>

  return %0 : tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1001692268, 0]]], [[[4864, 608, 1001692268, 0]]], [[[9728, 1216, 1001692268, 0]]], [[[14592, 1824, 1001692268, 0]]],
  // CHECK-SAME{LITERAL}:        [[[19456, 2432, 1001692268, 0]]], [[[24320, 3040, 1001692268, 0]]], [[[29184, 3648, 1001692268, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1001692268, 1001692268]]], [[[4864, 608, 1001692268, 1001692268]]], [[[9728, 1216, 1001692268, 1001692268]]],
  // CHECK-SAME{LITERAL}:        [[[14592, 1824, 1001692268, 1001692268]]], [[[19456, 2432, 1001692268, 1001692268]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [512, 4864, 1, 1] : tensor<512x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<512x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.0055118110030889511 : f64, prelu_alpha = [1.000000e+00], bias = 0.0055118110030889511 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 4864, 0, 0] [512, 4864, 1, 1] : tensor<512x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<512x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.0055118110030889511 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -1.250000e+20 : f64, clamp_high = 4.400000e+20 : f64,
  // CHECK-SAME:    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 0.1>
!qElemType1 = !quant.uniform<i8:f32, 0.055118110030889511>
!qElemType2 = !quant.uniform<i8:f16, 0.01:6>

// CHECK-LABEL:   @SplitNCEConvOverIC2ConvsWithScaledWeightsAndOutputZP
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:    [[WEIGHTS:%arg[0-9]]]: tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>
func.func @SplitNCEConvOverIC2ConvsWithScaledWeightsAndOutputZP(%arg0: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, %arg1: tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x512x4x1x!qElemType2, {order = #NHWC}> {
  // scale = 1057823284 = 0x3f0d1a34 = 0.5511811003088951 = 0.1 * 0.055118110030889511 / 0.01
  %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1057823284> : tensor<512x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64, scale = 0.55118110030889511 : f64, prelu_alpha = [1.000000e+00], bias = 0.5511811003088951 : f64, adder = 6.000000e+00 : f64>,
    rawFilterShape = [512, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, tensor<512x9728x1x1x!qElemType1, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1x!qElemType2, {order = #NHWC}>

  return %0 : tensor<1x512x4x1x!qElemType2, {order = #NHWC}>

  // scale = 1001692268 = 0x3bb49c6c = 0.005511811003088951f = 0.1 * 0.055118110030889511
  // CHECK-DAG:      [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1001692268, 0]]], [[[4864, 608, 1001692268, 0]]], [[[9728, 1216, 1001692268, 0]]], [[[14592, 1824, 1001692268, 0]]],
  // CHECK-SAME{LITERAL}:        [[[19456, 2432, 1001692268, 0]]], [[[24320, 3040, 1001692268, 0]]], [[[29184, 3648, 1001692268, 0]]]
  // CHECK-DAG:      [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1001692268, 1057823284]]], [[[4864, 608, 1001692268, 1057823284]]], [[[9728, 1216, 1001692268, 1057823284]]],
  // CHECK-SAME{LITERAL}:        [[[14592, 1824, 1001692268, 1057823284]]], [[[19456, 2432, 1001692268, 1057823284]]]

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [512, 4864, 1, 1] : tensor<512x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<512x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]], [[WEIGHTS_TABLE1]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.0055118110030889511 : f64, prelu_alpha = [1.000000e+00], bias = 0.55118110030889511 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 4864, 0, 0] [512, 4864, 1, 1] : tensor<512x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<512x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]], [[WEIGHTS_TABLE0]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   scale = 0.0055118110030889511 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [512, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME: -> tensor<1x512x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64,
  // CHECK-SAME:    scale = 99.999995944755397 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 6.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x512x4x1x!qElemType2, {order = #NHWC}>

  // CHECK:      return [[ADD_OUT1]] : tensor<1x512x4x1x!qElemType2, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 3.0>
!qElemType1 = !quant.uniform<i8:f32, 2.0>
!qElemType2 = !quant.uniform<i8:f16:1, {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16}>

// CHECK-LABEL:   @SplitNCEConvOverIC2ConvsWithPerChannelOutScale
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:    [[WEIGHTS:%arg[0-9]]]: tensor<16x9728x1x1x!qElemType1, {order = #NHWC}>
func.func @SplitNCEConvOverIC2ConvsWithPerChannelOutScale(%arg0: tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, %arg1: tensor<16x9728x1x1x!qElemType1, {order = #NHWC}>) -> tensor<1x16x4x1x!qElemType2, {order = #NHWC}> {
  %weights_table = const.Declare tensor<16x1x1x4xsi32>
    = dense<[[[[0, 0, 1031127695, 0]]], [[[11104, 704, 1039516303, 0]]], [[[22208, 1408, 1043878380, 0]]],
             [[[33312, 2112, 1047904911, 0]]], [[[44416, 2816, 1050253722, 0]]], [[[55520, 3520, 1052266988, 0]]],
             [[[66624, 4224, 1054280253, 0]]], [[[77728, 4928, 1056293519, 0]]], [[[88832, 5632, 1057635697, 0]]],
             [[[99936, 6336, 1058642330, 0]]], [[[111040, 7040, 1059648963, 0]]], [[[122144, 7744, 1060655596, 0]]],
             [[[133248, 8448, 1061662228, 0]]], [[[144352, 9152, 1062668861, 0]]], [[[155456, 9856, 1063675494, 0]]],
             [[[166560, 10560, 1064682127, 0]]]]> : tensor<16x1x1x4xsi32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %weights_table) {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 6.000000e+00 : f64>,
    rawFilterShape = [16, 9728, 1, 1],
    strides = [1, 1]
  } : tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, tensor<16x9728x1x1x!qElemType1, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  return %0 : tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  // CHECK:      [[DW_WT:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1008981770, 0]]], [[[32, 0, 1017370378, 0]]], [[[64, 0, 1022739088, 0]]],
  // CHECK-SAME{LITERAL}:        [[[96, 0, 1025758986, 0]]], [[[128, 0, 1028443341, 0]]], [[[160, 0, 1031127696, 0]]],
  // CHECK-SAME{LITERAL}:        [[[192, 0, 1032805417, 0]]], [[[224, 0, 1034147594, 0]]], [[[256, 0, 1035489772, 0]]],
  // CHECK-SAME{LITERAL}:        [[[288, 0, 1036831949, 0]]], [[[320, 0, 1038174127, 0]]], [[[352, 0, 1039516304, 0]]],
  // CHECK-SAME{LITERAL}:        [[[384, 0, 1040522936, 0]]], [[[416, 0, 1041194025, 0]]], [[[448, 0, 1041865113, 0]]],
  // CHECK-SAME{LITERAL}:        [[[480, 0, 1042536202, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK:      [[DW_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x1xf16>
  // CHECK:      [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1086324736, 0]]], [[[4864, 608, 1086324736, 0]]],
  // CHECK-SAME{LITERAL}:    [[[9728, 1216, 1086324736, 0]]], [[[14592, 1824, 1086324736, 0]]], [[[19456, 2432, 1086324736, 0]]], [[[24320, 3040, 1086324736, 0]]],
  // CHECK-SAME{LITERAL}:    [[[29184, 3648, 1086324736, 0]]], [[[34048, 4256, 1086324736, 0]]], [[[38912, 4864, 1086324736, 0]]], [[[43776, 5472, 1086324736, 0]]],
  // CHECK-SAME{LITERAL}:    [[[48640, 6080, 1086324736, 0]]], [[[53504, 6688, 1086324736, 0]]], [[[58368, 7296, 1086324736, 0]]], [[[63232, 7904, 1086324736, 0]]],
  // CHECK-SAME{LITERAL}:    [[[68096, 8512, 1086324736, 0]]], [[[72960, 9120, 1086324736, 0]]]]> : tensor<16x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4864, 1, 1] : tensor<16x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<16x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]], [[WEIGHTS_TABLE]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [16, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 4864, 0, 0] [16, 4864, 1, 1] : tensor<16x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<16x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]], [[WEIGHTS_TABLE]]) {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   rawFilterShape = [16, 4864, 1, 1], strides = [1, 1]}
  // CHECK-SAME: -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:    -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[DW_SCALE:%.+]] = VPU.NCE.DepthConvolution([[ADD_OUT1]], [[DW_WEIGHTS]], [[DW_WT]])
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64,
  // CHECK-SAME:    prelu_alpha = [1.000000e+00], adder = 6.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  // CHECK:      return [[DW_SCALE]] : tensor<1x16x4x1x!qElemType2, {order = #NHWC}>
}
