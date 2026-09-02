//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --ensure-nce-ops-size-requirements="enable-split-channel-for-dynamic-dequantize=true" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x16x16x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x16x16x!qElemType, {order = #NHWC}>) -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<36864x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<36864x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<36864x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [36864, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,

        strides = [1, 1]
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<36864x32x3x3x!qElemType2, {order = #NHWC}>, tensor<36864x1x1x4xsi32, {order = #NCHW}> -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<18432x1x1x4xsi32> = dense<10> : tensor<36864x1x1x4xsi32>, [#const.SubView<[18432, 0, 0, 0], [18432, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<18432x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<18432x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<36864x32x3x3xf16>
    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<18432x1x1x4xsi32> = dense<10> : tensor<36864x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [18432, 1, 1, 4]>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          -> tensor<1x18432x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          -> tensor<1x18432x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 18432, 0, 0]
    // CHECK-SAME:          -> tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x36864x16x16x!qElemType1, {order = #NHWC}>
}

// -----

// Checking tiling retry logic, will generate 252 tiles. For slice and conv, check the first two and last two, ignore others.
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

   //CHECK:    [[WEIGHTS_SLICE_FIRST:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_FIRST:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[CONV_FIRST:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_FIRST]], [[WEIGHTSTABLE_SLICE_FIRST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_1:%.+]] = VPU.Slice [[INPUT1]] [24576, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_1:%.+]] = VPU.Slice [[INPUT2]] [24576, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_1]], [[WEIGHTSTABLE_SLICE_1]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_250:%.+]] = VPU.Slice [[INPUT1]] [6144000, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_250:%.+]] = VPU.Slice [[INPUT2]] [6144000, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[CONV_250:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_250]], [[WEIGHTSTABLE_SLICE_250]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[WEIGHTS_SLICE_LAST:%.+]] = VPU.Slice [[INPUT1]] [6168576, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_LAST:%.+]] = VPU.Slice [[INPUT2]] [6168576, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[CONV_LAST:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS_SLICE_LAST]], [[WEIGHTSTABLE_SLICE_LAST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV_FIRST]],
   //CHECK-SAME:     [[CONV_1]]
   //CHECK-SAME:     [[CONV_250]]
   //CHECK-SAME:     [[CONV_LAST]])
   //CHECK-SAME:     -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    return  [[CONCAT:%.+]] tensor<1x6193152x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitExactDimensionLimitICNoConcatPattern
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x8192x256x4xf16, {order = #NHWC}>
// CHECK-SAME:    [[WEIGHTS:%arg[0-9]]]: tensor<4096x8192x1x1xf16, {order = #NHWC}>
func.func @SplitExactDimensionLimitICNoConcatPattern(%arg0: tensor<1x8192x256x4xf16, {order = #NHWC}>,
                                                     %arg1: tensor<4096x8192x1x1xf16, {order = #NHWC}>) -> tensor<1x4096x256x4xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [4096, 8192, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
  } : tensor<1x8192x256x4xf16, {order = #NHWC}>, tensor<4096x8192x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  return %0 : tensor<1x4096x256x4xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4096, 256, 4]
  // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [4096, 4096, 1, 1]
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[WEIGHTS_TILE0]]) rawFilterShape [4096, 4096, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 4096, 0, 0] [1, 4096, 256, 4]
  // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS]] [0, 4096, 0, 0] [4096, 4096, 1, 1]
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[WEIGHTS_TILE1]]) rawFilterShape [4096, 4096, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       return [[OUTPUT]] : tensor<1x4096x256x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitPaddedNCEConv
// CHECK-SAME:          [[INPUT:%.+]]: tensor<1x16416x1x1xf16, {order = #NHWC}>
func.func @SplitPaddedNCEConv(%arg0: tensor<1x16416x1x1xf16, {order = #NHWC}>) -> tensor<1x1472x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1472x16416x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<1472x1x1x4xsi32> = dense<10> : tensor<1472x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [1472, 16416, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        input_padding = [0, 0, 0, 0],
        output_padding = [0, 2, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,

        strides = [1, 1]
    } : tensor<1x16416x1x1xf16, {order = #NHWC}>, tensor<1472x16416x1x1xf16, {order = #NHWC}>, tensor<1472x1x1x4xsi32> -> tensor<1x1472x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x1472x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 5472, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TILE3:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 10944, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK-DAG:   [[WEIGHT_TABLE_TILE2:%.+]] = const.Declare tensor<1472x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHT_TABLE_TILE1:%.+]] = const.Declare tensor<1472x1x1x4xsi32>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS_TILE1]], [[WEIGHT_TABLE_TILE1]]) rawFilterShape [1472, 5472, 1, 1] {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS_TILE2]], [[WEIGHT_TABLE_TILE2]]) rawFilterShape [1472, 5472, 1, 1] {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[WEIGHTS_TILE3]], [[WEIGHT_TABLE_TILE2]]) rawFilterShape [1472, 5472, 1, 1] {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD1:%.+]] = VPU.NCE.Eltwise([[TILE1]], [[TILE2]]) {
    // CHECK-NOT:  mpe_engine
    // CHECK-SAME:    input_padding = [0, 2, 0, 0]
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK:       [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[TILE3]]) {
    // CHECK-NOT:  mpe_engine
    // CHECK-SAME:    input_padding = [0, 2, 0, 0]
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK:       return [[ADD2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @SplitMemPermuteAndDynamicDequantizeLargeIC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x14336x256x4xf16, {order = #NHWC}>
func.func @SplitMemPermuteAndDynamicDequantizeLargeIC(%arg0: tensor<1x14336x256x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x112x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x4096x112x1xf16> = dense<1.0> : tensor<1x4096x112x1xf16>

  %weights_mem_perm = VPU.MemPermute(%weights) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x112x4096x128x!qElemType> -> tensor<1x4096x112x128x!qElemType>
  %dequant = VPU.DynamicDequantize(%weights_mem_perm, %scales) {dstElemType = f16} : tensor<1x4096x112x128x!qElemType>, tensor<1x4096x112x1xf16> -> tensor<1x4096x112x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 14336, 1, 1]} : tensor<1x4096x112x128xf16> -> tensor<4096x14336x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<4096x14336x1x1xf16> -> tensor<4096x14336x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [4096, 14336, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x14336x256x4xf16, {order = #NHWC}>, tensor<4096x14336x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  %output_reshape = VPU.AffineReshape(%output) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 1024, 1]} : tensor<1x4096x256x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
  return %output_reshape : tensor<1x4096x1024x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x4096x38x1xf16> = dense<1.000000e+00> : tensor<1x4096x112x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 38, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x38x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 38, 4096, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x4096x38x1xf16> = dense<1.000000e+00> : tensor<1x4096x112x1xf16>, [#const.SubView<[0, 0, 38, 0], [1, 4096, 38, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x38x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.SubView<[0, 38, 0, 0], [1, 38, 4096, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x4096x36x1xf16> = dense<1.000000e+00> : tensor<1x4096x112x1xf16>, [#const.SubView<[0, 0, 76, 0], [1, 4096, 36, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x36x4096x128x!qElemType> = dense<1> : tensor<1x112x4096x128xi4>, [#const.SubView<[0, 76, 0, 0], [1, 36, 4096, 128]>, #const.CastElemType<!qElemType>]
  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4864, 256, 4]
  // CHECK-SAME:          to tensor<1x4864x256x4xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE_TILE0:%.+]] = VPU.MemPermute([[WEIGHTS_TILE0]])
  // CHECK-SAME:          -> tensor<1x4096x38x128x!qElemType>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[MEM_PERMUTE_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x38x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [4096, 4864, 1, 1]
  // CHECK-SAME:          -> tensor<4096x4864x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<4096x4864x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[PERM_TILE0]]) rawFilterShape [4096, 4864, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 4864, 0, 0] [1, 4864, 256, 4]
  // CHECK-SAME:          to tensor<1x4864x256x4xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE_TILE1:%.+]] = VPU.MemPermute([[WEIGHTS_TILE1]])
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[MEM_PERMUTE_TILE1]], [[SCALES_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x38x128xf16>
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [4096, 4864, 1, 1]
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[PERM_TILE1]]) rawFilterShape [4096, 4864, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 9728, 0, 0] [1, 4608, 256, 4]
  // CHECK-SAME:          to tensor<1x4608x256x4xf16, {order = #NHWC}>
  // CHECK:       [[MEM_PERMUTE_TILE2:%.+]] = VPU.MemPermute([[WEIGHTS_TILE2]])
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[MEM_PERMUTE_TILE2]], [[SCALES_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x36x128xf16>
  // CHECK:       [[RESHAPE_TILE2:%.+]] = VPU.AffineReshape([[DQ_TILE2]])
  // CHECK-SAME:          shape_value = [4096, 4608, 1, 1]
  // CHECK:       [[PERM_TILE2:%.+]] = VPU.PermuteCast([[RESHAPE_TILE2]])
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], [[PERM_TILE2]]) rawFilterShape [4096, 4608, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD0:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD1:%.+]] = VPU.NCE.Eltwise([[ADD0]], [[OUTPUT_TILE2]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       VPU.AffineReshape([[ADD1]])
  // CHECK-SAME:          shape_value = [1, 4096, 1024, 1]
  // CHECK-SAME:          -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @SplitDynamicDequantizeLargeIC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x24576x256x4xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeLargeIC(%arg0: tensor<1x24576x256x4xf16, {order = #NHWC}>) -> tensor<1x4096x1024x1xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x4096x192x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x4096x192x1xf16> = dense<1.0> : tensor<1x4096x192x1xf16>

  %dequant = VPU.DynamicDequantize(%weights, %scales) {dstElemType = f16} : tensor<1x4096x192x128x!qElemType>, tensor<1x4096x192x1xf16> -> tensor<1x4096x192x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [4096, 24576, 1, 1]} : tensor<1x4096x192x128xf16> -> tensor<4096x24576x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<4096x24576x1x1xf16> -> tensor<4096x24576x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast) rawFilterShape [4096, 24576, 1, 1] {
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x24576x256x4xf16, {order = #NHWC}>, tensor<4096x24576x1x1xf16, {order = #NHWC}> -> tensor<1x4096x256x4xf16, {order = #NHWC}>

  %output_reshape = VPU.AffineReshape(%output) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 4096, 1024, 1]} : tensor<1x4096x256x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
  return %output_reshape : tensor<1x4096x1024x1xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x4096x39x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 39, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x4096x39x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 4096, 39, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE4:%.+]] = const.Declare tensor<1x4096x36x1xf16> = dense<1.000000e+00> : tensor<1x4096x192x1xf16>, [#const.SubView<[0, 0, 156, 0], [1, 4096, 36, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE4:%.+]] = const.Declare tensor<1x4096x36x128x!qElemType> = dense<1> : tensor<1x4096x192x128xi4>, [#const.SubView<[0, 0, 156, 0], [1, 4096, 36, 128]>, #const.CastElemType<!qElemType>]
  // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 4992, 256, 4]
  // CHECK-SAME:          to tensor<1x4992x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x39x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [4096, 4992, 1, 1]
  // CHECK-SAME:          -> tensor<4096x4992x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<4096x4992x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_TILE0]], [[PERM_TILE0]]) rawFilterShape [4096, 4992, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 4992, 0, 0] [1, 4992, 256, 4]
  // CHECK-SAME:          to tensor<1x4992x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize({{%.+}}, {{%.+}}) {dstElemType = f16}
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [4096, 4992, 1, 1]
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_TILE1]], [[PERM_TILE1]]) rawFilterShape [4096, 4992, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 9984, 0, 0] [1, 4992, 256, 4]
  // CHECK-SAME:          to tensor<1x4992x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize({{%.+}}, {{%.+}}) {dstElemType = f16}
  // CHECK:       [[RESHAPE_TILE2:%.+]] = VPU.AffineReshape([[DQ_TILE2]])
  // CHECK-SAME:          shape_value = [4096, 4992, 1, 1]
  // CHECK:       [[PERM_TILE2:%.+]] = VPU.PermuteCast([[RESHAPE_TILE2]])
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_TILE2]], [[PERM_TILE2]]) rawFilterShape [4096, 4992, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 14976, 0, 0] [1, 4992, 256, 4]
  // CHECK-SAME:          to tensor<1x4992x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE3:%.+]] = VPU.DynamicDequantize({{%.+}}, {{%.+}}) {dstElemType = f16}
  // CHECK:       [[RESHAPE_TILE3:%.+]] = VPU.AffineReshape([[DQ_TILE3]])
  // CHECK-SAME:          shape_value = [4096, 4992, 1, 1]
  // CHECK:       [[PERM_TILE3:%.+]] = VPU.PermuteCast([[RESHAPE_TILE3]])
  // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[INPUT_TILE3]], [[PERM_TILE3]]) rawFilterShape [4096, 4992, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 19968, 0, 0] [1, 4608, 256, 4]
  // CHECK-SAME:          to tensor<1x4608x256x4xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE4:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE4]], [[SCALES_TILE4]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x4096x36x128xf16>
  // CHECK:       [[RESHAPE_TILE4:%.+]] = VPU.AffineReshape([[DQ_TILE4]])
  // CHECK-SAME:          shape_value = [4096, 4608, 1, 1]
  // CHECK:       [[PERM_TILE4:%.+]] = VPU.PermuteCast([[RESHAPE_TILE4]])
  // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.Convolution([[INPUT_TILE4]], [[PERM_TILE4]]) rawFilterShape [4096, 4608, 1, 1]
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD0:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD1:%.+]] = VPU.NCE.Eltwise([[ADD0]], [[OUTPUT_TILE2]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[OUTPUT_TILE3]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       [[ADD3:%.+]] = VPU.NCE.Eltwise([[ADD2]], [[OUTPUT_TILE4]])
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x4096x256x4xf16, {order = #NHWC}>
  // CHECK:       VPU.AffineReshape([[ADD3]])
  // CHECK-SAME:          shape_value = [1, 4096, 1024, 1]
  // CHECK-SAME:          -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitDynamicDequantizeMergedICAlignedToW
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x8320x128x8xf16, {order = #NHWC}>
func.func @SplitDynamicDequantizeMergedICAlignedToW(%arg0: tensor<1x8320x128x8xf16, {order = #NHWC}>) -> tensor<1x3072x128x8xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<1x3072x65x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.CastElemType<!qElemType>]
  %scales = const.Declare tensor<1x3072x65x1xf16> = dense<1.0> : tensor<1x3072x65x1xf16>
  %weights_table = const.Declare tensor<3072x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<3072x1x1x4xsi32>

  %dequant = VPU.DynamicDequantize(%weights, %scales) {dstElemType = f16} : tensor<1x3072x65x128x!qElemType>, tensor<1x3072x65x1xf16> -> tensor<1x3072x65x128xf16>
  %weights_reshape = VPU.AffineReshape(%dequant) {dim_mapping = [[0], [0], [1], [1, 2, 3]], shape_value = [3072, 8320, 1, 1]} : tensor<1x3072x65x128xf16> -> tensor<3072x8320x1x1xf16>
  %weights_permute_cast = VPU.PermuteCast(%weights_reshape) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<3072x8320x1x1xf16> -> tensor<3072x8320x1x1xf16, {order = #NHWC}>

  %output = VPU.NCE.Convolution(%arg0, %weights_permute_cast, %weights_table) rawFilterShape [3072, 8320, 1, 1] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : tensor<1x8320x128x8xf16, {order = #NHWC}>, tensor<3072x8320x1x1xf16, {order = #NHWC}>, tensor<3072x1x1x4xsi32, {order = #NCHW}> -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  return %output : tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK-DAG:   [[SCALES_TILE0:%.+]] = const.Declare tensor<1x3072x22x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 3072, 22, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE0:%.+]] = const.Declare tensor<1x3072x22x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 0, 0], [1, 3072, 22, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE1:%.+]] = const.Declare tensor<1x3072x22x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 22, 0], [1, 3072, 22, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1x3072x22x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 22, 0], [1, 3072, 22, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   [[SCALES_TILE2:%.+]] = const.Declare tensor<1x3072x21x1xf16> = dense<1.000000e+00> : tensor<1x3072x65x1xf16>, [#const.SubView<[0, 0, 44, 0], [1, 3072, 21, 1]>]
  // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1x3072x21x128x!qElemType> = dense<1> : tensor<1x3072x65x128xi4>, [#const.SubView<[0, 0, 44, 0], [1, 3072, 21, 128]>, #const.CastElemType<!qElemType>]
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>
  // CHECK-DAG:   const.Declare tensor<3072x1x1x4xsi32>

  // CHECK:       [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2816, 128, 8]
  // CHECK-SAME:          to tensor<1x2816x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE0:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE0]], [[SCALES_TILE0]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x22x128xf16>
  // CHECK:       [[RESHAPE_TILE0:%.+]] = VPU.AffineReshape([[DQ_TILE0]])
  // CHECK-SAME:          shape_value = [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16>
  // CHECK:       [[PERM_TILE0:%.+]] = VPU.PermuteCast([[RESHAPE_TILE0]])
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0]], [[PERM_TILE0]], {{%.+}}) rawFilterShape [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2816, 0, 0] [1, 2816, 128, 8]
  // CHECK-SAME:          to tensor<1x2816x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE1:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE1]], [[SCALES_TILE1]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x22x128xf16>
  // CHECK:       [[RESHAPE_TILE1:%.+]] = VPU.AffineReshape([[DQ_TILE1]])
  // CHECK-SAME:          shape_value = [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16>
  // CHECK:       [[PERM_TILE1:%.+]] = VPU.PermuteCast([[RESHAPE_TILE1]])
  // CHECK-SAME:          -> tensor<3072x2816x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1]], [[PERM_TILE1]], {{%.+}}) rawFilterShape [3072, 2816, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 5632, 0, 0] [1, 2688, 128, 8]
  // CHECK-SAME:          to tensor<1x2688x128x8xf16, {order = #NHWC}>
  // CHECK:       [[DQ_TILE2:%.+]] = VPU.DynamicDequantize([[WEIGHTS_TILE2]], [[SCALES_TILE2]]) {dstElemType = f16}
  // CHECK-SAME:          -> tensor<1x3072x21x128xf16>
  // CHECK:       [[RESHAPE_TILE2:%.+]] = VPU.AffineReshape([[DQ_TILE2]])
  // CHECK-SAME:          shape_value = [3072, 2688, 1, 1]
  // CHECK-SAME:          -> tensor<3072x2688x1x1xf16>
  // CHECK:       [[PERM_TILE2:%.+]] = VPU.PermuteCast([[RESHAPE_TILE2]])
  // CHECK-SAME:          -> tensor<3072x2688x1x1xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2]], [[PERM_TILE2]], {{%.+}}) rawFilterShape [3072, 2688, 1, 1]
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>

  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT_TILE0]], [[OUTPUT_TILE1]]) {op_type = #VPU.eltwise_type<ADD>
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>
  // CHECK:       [[FINAL_OUTPUT:%.+]] = VPU.NCE.Eltwise([[OUTPUT]], [[OUTPUT_TILE2]]) {op_type = #VPU.eltwise_type<ADD>
  // CHECK-NOT:   mpe_engine
  // CHECK-SAME:          -> tensor<1x3072x128x8xf16, {order = #NHWC}>
  // CHECK:       return [[FINAL_OUTPUT]] : tensor<1x3072x128x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 100000 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @DoNotApplyPipelinableICSplitToSparseConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x8x8xf16, {order = #NHWC}>
func.func @DoNotApplyPipelinableICSplitToSparseConv(%arg0: tensor<1x256x8x8xf16, {order = #NHWC}>) -> tensor<1x64x8x8xf16, {order = #NHWC}> {
  %sparsity_map = const.Declare tensor<1x256x8x8xi1, {order = #NHWC}> = dense<true> : tensor<1x256x8x8xi1>, [#const.Reorder<#NHWC>]
  %se_table = VPU.StorageElementTable { dataElemType = f16, dataShape = [1, 256, 8, 8], seDepth = 1 : i64, seSize = [256] } -> tensor<1x1x8x8xi32, {order = #NHWC}>
  %sparse_input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %se_table) -> !VPU.SparseTensor<data=tensor<1x256x8x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x8x8xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>>
  %weights = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<64x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<64x1x1x4xsi32>
  %output = VPU.NCE.Convolution(%sparse_input, %weights, %weights_table) rawFilterShape [64, 256, 1, 1] {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    resultSegmentSizes = array<i32: 1, 0, 0, 0>,
    strides = [1, 1]
  } : !VPU.SparseTensor<data=tensor<1x256x8x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x8x8xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>>, tensor<64x256x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32, {order = #NCHW}> -> tensor<1x64x8x8xf16, {order = #NHWC}>

  return %output : tensor<1x64x8x8xf16, {order = #NHWC}>

  // CHECK:       [[SE_TABLE:%.+]] = VPU.StorageElementTable
  // CHECK-SAME:          dataShape = [1, 256, 8, 8]
  // CHECK:       [[SPARSE_INPUT:%.+]] = VPU.GroupSparseTensor([[INPUT]], {{%.+}}, [[SE_TABLE]])
  // CHECK-SAME:          -> !VPU.SparseTensor<data=tensor<1x256x8x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x8x8xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>>
  // CHECK-NOT:   VPU.Slice [[SPARSE_INPUT]]
  // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[SPARSE_INPUT]], {{%.+}}, {{%.+}}) rawFilterShape [64, 256, 1, 1]
  // CHECK-SAME:          : !VPU.SparseTensor<data=tensor<1x256x8x8xf16, {order = #NHWC}>, sparsity_map=tensor<1x256x8x8xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x8x8xi32, {order = #NHWC}>>, tensor<64x256x1x1xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32, {order = #NCHW}> -> tensor<1x64x8x8xf16, {order = #NHWC}>
  // CHECK:       return [[OUTPUT]] : tensor<1x64x8x8xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

config.Resources 3 of @NCE at 2.100000e+03 MHz {
  config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitWithPerTensorStaticScale
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x7680x1920x4xf16, {order = #NHWC}>
func.func @SplitWithPerTensorStaticScale(%arg0: tensor<1x7680x1920x4xf16, {order = #NHWC}>) -> tensor<1x48x1920x4xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<48x7680x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<48x7680x1x1xf16, {order = #NHWC}>

  %weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1065353216> : tensor<48x1x1x4xsi32>

  %output = VPU.NCE.Convolution(%arg0, %weights, %weights_table) rawFilterShape [48, 7680, 1, 1] {
    input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, 
    output_padding = [0, 8, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, 
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 0.01141088661469096 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, 
    resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
  } : tensor<1x7680x1920x4xf16, {order = #NHWC}>, tensor<48x7680x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x1920x4xf16, {order = #NHWC}>

  return %output : tensor<1x48x1920x4xf16, {order = #NHWC}>

  // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<48x3840x1x1xf16, {order = #NHWC}>
  // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<48x3840x1x1xf16, {order = #NHWC}>
  // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<48x1x1x4xsi32>
  // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<48x1x1x4xsi32>
  // CHECK: [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3840, 1920, 4] : tensor<1x7680x1920x4xf16, {order = #NHWC}> to tensor<1x3840x1920x4xf16, {order = #NHWC}>
  // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[CST_1]], [[CST_3]])
  // CHECK-SAME: scale = 1.000000e+00 : f64
  // CHECK: [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 3840, 0, 0] [1, 3840, 1920, 4] : tensor<1x7680x1920x4xf16, {order = #NHWC}> to tensor<1x3840x1920x4xf16, {order = #NHWC}>
  // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[CST_0]], [[CST_2]])
  // CHECK-SAME: scale = 1.000000e+00 : f64
  // CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_0]], [[CONV_1]])
  // CHECK-NOT: mpe_engine
  // CHECK-SAME: scale = 0.01141088661469096 : f64
  // CHECK: return [[ELTWISE]] : tensor<1x48x1920x4xf16, {order = #NHWC}>
}
