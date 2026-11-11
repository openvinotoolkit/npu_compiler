//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --ensure-nce-ops-size-requirements --canonicalize --mlir-elide-elementsattrs-if-larger 64 %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOCSplitWeightTableFormat
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOCSplitWeightTableFormat(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<36864x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %scale_table = const.Declare tensor<36864x1x1x1xf32, {order = #NCHW}> = dense<1.0> : tensor<36864x1x1x1xf32>
    %bias_table = const.Declare tensor<36864x1x1x1xf32, {order = #NCHW}> = dense<2.0> : tensor<36864x1x1x1xf32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %scale_table, %bias_table) rawFilterShape [36864, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        
        strides = [1, 1],
        operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<36864x32x3x3x!qElemType2, {order = #NHWC}>, tensor<36864x1x1x1xf32, {order = #NCHW}>, tensor<36864x1x1x1xf32, {order = #NCHW}> -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[SCALE_TABLE_TILE1:%.+]] = const.Declare tensor<18432x1x1x1xf32> = dense<1.000000e+00> : tensor<36864x1x1x1xf32>, [#const.SubView<[18432, 0, 0, 0], [18432, 1, 1, 1]>]
    // CHECK-DAG:        [[BIAS_TABLE_TILE1:%.+]] = const.Declare tensor<18432x1x1x1xf32> = dense<2.000000e+00> : tensor<36864x1x1x1xf32>, [#const.SubView<[18432, 0, 0, 0], [18432, 1, 1, 1]>]
    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<18432x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<18432x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>
    // CHECK-DAG:        [[SCALE_TABLE_TILE0:%.+]] = const.Declare tensor<18432x1x1x1xf32> = dense<1.000000e+00> : tensor<36864x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [18432, 1, 1, 1]>]
    // CHECK-DAG:        [[BIAS_TABLE_TILE0:%.+]] = const.Declare tensor<18432x1x1x1xf32> = dense<2.000000e+00> : tensor<36864x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [18432, 1, 1, 1]>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[SCALE_TABLE_TILE0]], [[BIAS_TABLE_TILE0]])
    // CHECK-SAME:          -> tensor<1x18432x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[SCALE_TABLE_TILE1]], [[BIAS_TABLE_TILE1]])
    // CHECK-SAME:          -> tensor<1x18432x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 18432, 0, 0]
    // CHECK-SAME:          -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>
}

// -----

// Checking tiling retry logic, will generate 189 tiles. For slice and conv, check the first two and last two, ignore others.
// For concat, only check the first and last input, ignore others
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @CheckTilingRetryLogic
// CHECK-SAME:    [[INPUT0:%arg[0-9]]]: tensor<1x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:    [[INPUT1:%arg[0-9]]]: tensor<6193152x16x1x1xf16, {order = #NHWC}>
// CHECK-SAME:    [[INPUT2:%arg[0-9]]]: tensor<6193152x1x1x4xsi32, {order = #NCHW}>
func.func @CheckTilingRetryLogic(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>,
                                %arg1: tensor<6193152x16x1x1xf16, {order = #NHWC}>,
                                %arg2: tensor<6193152x1x1x4xsi32, {order = #NCHW}>) -> tensor<1x6193152x1x1xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) rawFilterShape [6193152, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        
        strides = [1, 1]} : tensor<1x16x1x1xf16, {order = #NHWC}>, tensor<6193152x16x1x1xf16, {order = #NHWC}>, tensor<6193152x1x1x4xsi32, {order = #NCHW}> -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_FIRST:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [32768, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<32768x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_FIRST:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [32768, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<32768x1x1x4xsi32>
   //CHECK:    [[CONV_FIRST:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_FIRST]], [[WEIGHTSTABLE_SLICE_FIRST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x32768x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_1:%.+]] = VPU.Slice [[INPUT1]] [32768, 0, 0, 0] [32768, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<32768x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_1:%.+]] = VPU.Slice [[INPUT2]] [32768, 0, 0, 0] [32768, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<32768x1x1x4xsi32>
   //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_1]], [[WEIGHTSTABLE_SLICE_1]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x32768x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_187:%.+]] = VPU.Slice [[INPUT1]] [6127616, 0, 0, 0] [32768, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<32768x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_187:%.+]] = VPU.Slice [[INPUT2]] [6127616, 0, 0, 0] [32768, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<32768x1x1x4xsi32>
   //CHECK:    [[CONV_187:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_187]], [[WEIGHTSTABLE_SLICE_187]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x32768x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_LAST:%.+]] = VPU.Slice [[INPUT1]] [6160384, 0, 0, 0] [32768, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<32768x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_LAST:%.+]] = VPU.Slice [[INPUT2]] [6160384, 0, 0, 0] [32768, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<32768x1x1x4xsi32>
   //CHECK:    [[CONV_LAST:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_LAST]], [[WEIGHTSTABLE_SLICE_LAST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x32768x1x1xf16, {order = #NHWC}>

   //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV_FIRST]],
   //CHECK-SAME:     [[CONV_1]]
   //CHECK-SAME:     [[CONV_187]]
   //CHECK-SAME:     [[CONV_LAST]]
   //CHECK-SAME:     -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    return  [[CONCAT:%.+]] tensor<1x6193152x1x1xf16, {order = #NHWC}>
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
  %scale_table = const.Declare tensor<16x1x1x1xf32> = dense<[[[[0.06]]], [[[0.12]]], [[[0.18]]], [[[0.24]]], [[[0.3]]], [[[0.36]]], [[[0.42]]],
                                                             [[[0.48]]], [[[0.54]]], [[[0.6]]], [[[0.66]]], [[[0.72]]], [[[0.78]]], [[[0.84]]],
                                                             [[[0.9]]], [[[0.96]]]]> : tensor<16x1x1x1xf32>
  %bias_table = const.Declare tensor<16x1x1x1xf32> = dense<2.0> : tensor<16x1x1x1xf32>
  %0 = VPU.NCE.Convolution(%arg0, %arg1, %scale_table, %bias_table) rawFilterShape [16, 9728, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64, prelu_alpha = [1.000000e+00], adder = 6.000000e+00 : f64>,
    
    strides = [1, 1],
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
  } : tensor<1x9728x4x1x!qElemType, {order = #NHWC}>, tensor<16x9728x1x1x!qElemType1, {order = #NHWC}>, tensor<16x1x1x1xf32>, tensor<16x1x1x1xf32>
    -> tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  return %0 : tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  // CHECK:      [[DW_WT:%.+]] = const.Declare tensor<16x1x1x4xsi32>
  // CHECK-SAME{LITERAL}:  dense<[[[[0, 0, 1008981770, 0]]], [[[32, 0, 1017370378, 0]]], [[[64, 0, 1022739088, 0]]],
  // CHECK-SAME{LITERAL}:        [[[96, 0, 1025758986, 0]]], [[[128, 0, 1028443341, 0]]], [[[160, 0, 1031127696, 0]]],
  // CHECK-SAME{LITERAL}:        [[[192, 0, 1032805417, 0]]], [[[224, 0, 1034147594, 0]]], [[[256, 0, 1035489772, 0]]],
  // CHECK-SAME{LITERAL}:        [[[288, 0, 1036831949, 0]]], [[[320, 0, 1038174127, 0]]], [[[352, 0, 1039516304, 0]]],
  // CHECK-SAME{LITERAL}:        [[[384, 0, 1040522936, 0]]], [[[416, 0, 1041194025, 0]]], [[[448, 0, 1041865113, 0]]],
  // CHECK-SAME{LITERAL}:        [[[480, 0, 1042536202, 0]]]]> : tensor<16x1x1x4xsi32>
  // CHECK:      [[DW_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x1x1xf16>
  // CHECK:      [[BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
  // CHECK:      [[SCALE_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<6.000000e+00> : tensor<16x1x1x1xf32>
  // CHECK:      [[ZERO_BIAS_TABLE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00> : tensor<16x1x1x1xf32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [16, 4864, 1, 1] : tensor<16x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<16x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0]], [[SCALE_TABLE]], [[BIAS_TABLE]]) rawFilterShape [16, 4864, 1, 1] {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME:   -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 4, 1] : tensor<1x9728x4x1x!qElemType, {order = #NHWC}> to tensor<1x4864x4x1x!qElemType, {order = #NHWC}>
  // CHECK:      [[WEIGHTS_SLICE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 4864, 0, 0] [16, 4864, 1, 1] : tensor<16x9728x1x1x!qElemType1, {order = #NHWC}> to tensor<16x4864x1x1x!qElemType1, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1]], [[SCALE_TABLE]], [[ZERO_BIAS_TABLE]]) rawFilterShape [16, 4864, 1, 1] {
  // CHECK-SAME:   ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:   prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:   strides = [1, 1]}
  // CHECK-SAME: -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0]], [[CONV_OUT1]]) {
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>,
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
  // CHECK-SAME:          prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
  // CHECK-SAME:    -> tensor<1x16x4x1xf16, {order = #NHWC}>

  // CHECK:      [[DW_SCALE:%.+]] = VPU.NCE.DepthConvolution([[ADD_OUT1]], [[DW_WEIGHTS]], [[DW_WT]])
  // CHECK-NOT:  mpe_engine
  // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.340000e+02 : f64, clamp_high = 1.210000e+02 : f64,
  // CHECK-SAME:    prelu_alpha = [1.000000e+00], adder = 6.000000e+00 : f64>
  // CHECK-SAME: } -> tensor<1x16x4x1x!qElemType2, {order = #NHWC}>

  // CHECK:      return [[DW_SCALE]] : tensor<1x16x4x1x!qElemType2, {order = #NHWC}>
}
