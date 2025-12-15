//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [36864, 32, 3, 3],
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

// Checking tiling retry logic, will generate 189 tiles. For slice and depthconv, check the first two and last two, ignor others.
// For concat, only check the first and last input, ignor others
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @CheckTilingRetryLogic
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x6193152x1x1xf16, {order = #NHWC}>
func.func @CheckTilingRetryLogic(%arg0: tensor<1x6193152x1x1xf16, {order = #NHWC}>,
                                %arg1: tensor<6193152x16x1x1xf16, {order = #NHWC}>,
                                %arg2: tensor<6193152x1x1x4xsi32, {order = #NCHW}>) -> tensor<1x6193152x1x1xf16, {order = #NHWC}> {
  %0 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [6193152, 1, 1, 1],
        strides = [1, 1]} -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_FIRST:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 24576, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x24576x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_FIRST:%.+]] = VPU.Slice %arg1 [0, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_FIRST:%.+]] = VPU.Slice %arg2 [0, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_FIRST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_FIRST]], [[WEIGHTS_SLICE_FIRST]], [[WEIGHTSTABLE_SLICE_FIRST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_1:%.+]]  = VPU.Slice %arg0 [0, 24576, 0, 0] [1, 24576, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x24576x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_1:%.+]] = VPU.Slice %arg1 [24576, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_1:%.+]] = VPU.Slice %arg2 [24576, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_1:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_1]], [[WEIGHTS_SLICE_1]], [[WEIGHTSTABLE_SLICE_1]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_187:%.+]] = VPU.Slice %arg0 [0, 147456, 0, 0] [1, 24576, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x24576x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_187:%.+]] = VPU.Slice %arg1 [147456, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_187:%.+]] = VPU.Slice %arg2 [147456, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_187:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_187]], [[WEIGHTS_SLICE_187]], [[WEIGHTSTABLE_SLICE_187]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_LAST:%.+]] = VPU.Slice %arg0 [0, 172032, 0, 0] [1, 24576, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x24576x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_LAST:%.+]] = VPU.Slice %arg1 [172032, 0, 0, 0] [24576, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<24576x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_LAST:%.+]] = VPU.Slice %arg2 [172032, 0, 0, 0] [24576, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<24576x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_LAST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_LAST]], [[WEIGHTS_SLICE_LAST]], [[WEIGHTSTABLE_SLICE_LAST]])
   //CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
   //CHECK-SAME:              -> tensor<1x24576x1x1xf16, {order = #NHWC}>

   //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV_FIRST]],
   //CHECK-NOT:      DEPTHCONV_1
   //CHECK-NOT:      DEPTHCONV_187
   //CHECK:         [[DEPTHCONV_LAST]]
   //CHECK-SAME:     -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    return  [[CONCAT:%.+]] tensor<1x6193152x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitPaddedNCEConv
// CHECK-SAME:          [[INPUT:%.+]]: tensor<1x16416x1x1xf16, {order = #NHWC}>
func.func @SplitPaddedNCEConv(%arg0: tensor<1x16416x1x1xf16, {order = #NHWC}>) -> tensor<1x1472x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1472x16416x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<1472x1x1x4xsi32> = dense<10> : tensor<1472x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        input_padding = [0, 0, 0, 0],
        output_padding = [0, 2, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
        rawFilterShape = [1472, 16416, 1, 1],
        strides = [1, 1]
    } : tensor<1x16416x1x1xf16, {order = #NHWC}>, tensor<1472x16416x1x1xf16, {order = #NHWC}>, tensor<1472x1x1x4xsi32> -> tensor<1x1472x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x1472x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_TILE1:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TILE2:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 5472, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TILE3:%.+]] = const.Declare tensor<1472x5472x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1472x16416x1x1xf16>, [#const.SubView<[0, 10944, 0, 0], [1472, 5472, 1, 1]>, #const.Reorder<#NHWC>]

    // CHECK-DAG:   [[WEIGHT_TABLE_TILE2:%.+]] = const.Declare tensor<1472x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHT_TABLE_TILE1:%.+]] = const.Declare tensor<1472x1x1x4xsi32>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS_TILE1]], [[WEIGHT_TABLE_TILE1]]) {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 5472, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS_TILE2]], [[WEIGHT_TABLE_TILE2]]) {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 10944, 0, 0] [1, 5472, 1, 1]
    // CHECK:       [[TILE3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[WEIGHTS_TILE3]], [[WEIGHT_TABLE_TILE2]]) {
    // CHECK-NOT:     input_padding
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK-SAME:    -> tensor<1x1472x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ADD1:%.+]] = VPU.NCE.Eltwise([[TILE1]], [[TILE2]]) {
    // CHECK-SAME:    input_padding = [0, 2, 0, 0]
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK:       [[ADD2:%.+]] = VPU.NCE.Eltwise([[ADD1]], [[TILE3]]) {
    // CHECK-SAME:    input_padding = [0, 2, 0, 0]
    // CHECK-SAME:    output_padding = [0, 2, 0, 0]
    // CHECK:       return [[ADD2]]
}
