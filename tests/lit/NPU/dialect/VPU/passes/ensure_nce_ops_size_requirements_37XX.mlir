//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --canonicalize --mlir-elide-elementsattrs-if-larger 64 %s | FileCheck %s
// REQUIRES: arch-NPU37XX

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
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [9216, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x16x16x!qElemType, {order = #NHWC}>, tensor<9216x32x3x3x!qElemType2, {order = #NHWC}>, tensor<9216x1x1x4xsi32, {order = #NCHW}> -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>, [#const.SubView<[4608, 0, 0, 0], [4608, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[4608, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<4608x1x1x4xsi32> = dense<10> : tensor<9216x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [4608, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<4608x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<9216x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [4608, 32, 3, 3]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          -> tensor<1x4608x16x16x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 4608, 0, 0]
    // CHECK-SAME:          -> tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x9216x16x16x!qElemType1, {order = #NHWC}>
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
    ppe = #VPU.PPEStub<>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    rawFilterShape = [6193152, 1, 1, 1],
    strides = [1, 1]} -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

  return %0 : tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_FIRST:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_FIRST:%.+]] = VPU.Slice %arg1 [0, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_FIRST:%.+]] = VPU.Slice %arg2 [0, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_FIRST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_FIRST]], [[WEIGHTS_SLICE_FIRST]], [[WEIGHTSTABLE_SLICE_FIRST]])
   //CHECK-SAME:               -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_1:%.+]]  = VPU.Slice %arg0 [0, 8192, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_1:%.+]] = VPU.Slice %arg1 [8192, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_1:%.+]] = VPU.Slice %arg2 [8192, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_1:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_1]], [[WEIGHTS_SLICE_1]], [[WEIGHTSTABLE_SLICE_1]])
   //CHECK-SAME:               -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_754:%.+]] = VPU.Slice %arg0 [0, 6176768, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_754:%.+]] = VPU.Slice %arg1 [6176768, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_754:%.+]] = VPU.Slice %arg2 [6176768, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_754:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_754]], [[WEIGHTS_SLICE_754]], [[WEIGHTSTABLE_SLICE_754]])
   //CHECK-SAME:               -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[ACT_SLICE_LAST:%.+]] = VPU.Slice %arg0 [0, 6184960, 0, 0] [1, 8192, 1, 1] : tensor<1x6193152x1x1xf16, {order = #NHWC}> to tensor<1x8192x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTS_SLICE_LAST:%.+]] = VPU.Slice %arg1 [6184960, 0, 0, 0] [8192, 16, 1, 1] : tensor<6193152x16x1x1xf16, {order = #NHWC}> to tensor<8192x16x1x1xf16, {order = #NHWC}>
   //CHECK:    [[WEIGHTSTABLE_SLICE_LAST:%.+]] = VPU.Slice %arg2 [6184960, 0, 0, 0] [8192, 1, 1, 4] : tensor<6193152x1x1x4xsi32, {order = #NCHW}> to tensor<8192x1x1x4xsi32>
   //CHECK:    [[DEPTHCONV_LAST:%.+]] = VPU.NCE.DepthConvolution([[ACT_SLICE_LAST]], [[WEIGHTS_SLICE_LAST]], [[WEIGHTSTABLE_SLICE_LAST]])
   //CHECK-SAME:                -> tensor<1x8192x1x1xf16, {order = #NHWC}>

   //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[DEPTHCONV_FIRST]],
   //CHECK-NOT:      DEPTHCONV_1
   //CHECK-NOT:      DEPTHCONV_754
   //CHECK-SAME:     [[DEPTHCONV_LAST]])
   //CHECK-SAME:     -> tensor<1x6193152x1x1xf16, {order = #NHWC}>

   //CHECK:    return  [[CONCAT:%.+]] tensor<1x6193152x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NCEPermuteLargeWidth
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x3x32x8208xf16>
func.func @NCEPermuteLargeWidth(%arg0: tensor<1x3x32x8208xf16>) -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>
    } -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    return %nce_permute : tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 32, 4112] :
    // CHECK-SAME:      tensor<1x3x32x8208xf16> to tensor<1x3x32x4112xf16>

    // CHECK:       [[FIRST_NCE_PERM:%.*]] = VPU.NCE.Permute([[FIRST_SLICE]])
    // CHECK-SAME:      -> tensor<1x4x32x4112x!qElemType, {order = #NHWC}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 4112] [1, 3, 32, 4096] :
    // CHECK-SAME:      tensor<1x3x32x8208xf16> to tensor<1x3x32x4096xf16>

    // CHECK:       [[SECOND_NCE_PERM:%.*]] = VPU.NCE.Permute([[SECOND_SLICE]])
    // CHECK-SAME:      -> tensor<1x4x32x4096x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_NCE_PERM]], [[SECOND_NCE_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 0, 4112]]}
    // CHECK-SAME:      : tensor<1x4x32x4112x!qElemType, {order = #NHWC}>, tensor<1x4x32x4096x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x4x32x8208x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x4x32x8208x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @SOKNCEPermuteLargeChannels
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x8204x32x32xf16>
func.func @SOKNCEPermuteLargeChannels(%arg0: tensor<1x8204x32x32xf16>) -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 8208 : i64
    } -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    return %nce_permute : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4104, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4104x32x32xf16>

    // CHECK:       [[FIRST_NCE_PERM:%.*]] = VPU.NCE.Permute([[FIRST_SLICE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 4104, 0, 0] [1, 4100, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4100x32x32xf16>

    // CHECK:       [[SECOND_NCE_PERM:%.*]] = VPU.NCE.Permute([[SECOND_SLICE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_NCE_PERM]], [[SECOND_NCE_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 4104, 0, 0]]}
    // CHECK-SAME:      : tensor<1x4104x32x32x!qElemType, {order = #NHWC}>, tensor<1x4104x32x32x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NCEPermuteLargeChannels
// CHECK-SAME:        [[INPUT:%.+]]: tensor<1x8204x32x32xf16>
func.func @NCEPermuteLargeChannels(%arg0: tensor<1x8204x32x32xf16>) -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}> {
    %nce_permute = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 8208 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>
    } -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    return %nce_permute : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[FIRST_SLICE:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4104, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4104x32x32xf16>

    // CHECK:       [[FIRST_NCE_PERM:%.*]] = VPU.NCE.Permute([[FIRST_SLICE]])
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4104 : i64,
    // CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>}
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[SECOND_SLICE:%.*]] = VPU.Slice %arg0 [0, 4104, 0, 0] [1, 4100, 32, 32] :
    // CHECK-SAME:      tensor<1x8204x32x32xf16> to tensor<1x4100x32x32xf16>

    // CHECK:       [[SECOND_NCE_PERM:%.*]] = VPU.NCE.Permute([[SECOND_SLICE]])
    // CHECK-SAME:      {dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4104 : i64,
    // CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>}
    // CHECK-SAME:      -> tensor<1x4104x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[FIRST_NCE_PERM]], [[SECOND_NCE_PERM]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 4104, 0, 0]]}
    // CHECK-SAME:      : tensor<1x4104x32x32x!qElemType, {order = #NHWC}>, tensor<1x4104x32x32x!qElemType, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x8208x32x32x!qElemType, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x8208x32x32x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @Int4ConvolutionLargeChannels
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x18944x1x1xf16, {order = #NHWC}>
func.func @Int4ConvolutionLargeChannels(%arg0: tensor<1x18944x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x1xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %weights = const.Declare tensor<32x18944x1x1x!quant.uniform<i4:f16, 0.003971099853515625>, {order = #NHWC}> = dense<1.000000e+00> :
                tensor<32x18944x1x1xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui4>, #const.CastElemType<!quant.uniform<u4:f16, 0.003971099853515625:8>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 0.003971099853515625 : f64>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [32, 18944, 1, 1], strides = [1, 1]
    } : tensor<1x18944x1x1xf16, {order = #NHWC}>, tensor<32x18944x1x1x!quant.uniform<i4:f16, 0.003971099853515625>, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_0:%.+]] = const.Declare tensor<32x6272x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_1:%.+]] = const.Declare tensor<32x6336x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK:       [[WEIGHTS_2:%.+]] = const.Declare tensor<32x6336x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[WEIGHTSTABLE_0:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[WEIGHTSTABLE_1:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[WEIGHTSTABLE_2:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:       [[INSLICE_0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 6336, 1, 1] : tensor<1x18944x1x1xf16, {order = #NHWC}> to tensor<1x6336x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[INSLICE_0]], [[WEIGHTS_2]], [[WEIGHTSTABLE_2]]) {
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       [[INSLICE_1:%.+]] = VPU.Slice [[INPUT0]] [0, 6336, 0, 0] [1, 6336, 1, 1] : tensor<1x18944x1x1xf16, {order = #NHWC}> to tensor<1x6336x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INSLICE_1]], [[WEIGHTS_1]], [[WEIGHTSTABLE_1]]) {
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       [[INSLICE_2:%.+]] = VPU.Slice [[INPUT0]] [0, 12672, 0, 0] [1, 6272, 1, 1] : tensor<1x18944x1x1xf16, {order = #NHWC}> to tensor<1x6272x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CONV_2:%.+]] = VPU.NCE.Convolution([[INSLICE_2]], [[WEIGHTS_0]], [[WEIGHTSTABLE_0]]) {
    // CHECK-SAME:  }
    // CHECK-SAME:  -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       [[ADD_0:%.+]] = VPU.NCE.Eltwise([[CONV_0]], [[CONV_1]]) {
    // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:  } -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       [[ADD_1:%.+]] = VPU.NCE.Eltwise([[ADD_0]], [[CONV_2]]) {
    // CHECK-SAME:    op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:  } -> tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:       return   [[ADD_1]] : tensor<1x32x1x1xf16, {order = #NHWC}>
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
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
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
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER4:%.+]], [[WEIGHTS_TABLE4:%.+]]) {
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT0:%.+]] = VPU.Concat([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE1:%.+]] = VPU.Slice %arg0 [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER3:%.+]], [[WEIGHTS_TABLE3:%.+]]) {
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}
  // CHECK:       [[CONV_OUT3:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]]) {
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT1:%.+]] = VPU.Concat([[CONV_OUT2:%.+]], [[CONV_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE2:%.+]] = VPU.Slice %arg0 [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT4:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]]) {
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONV_OUT5:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]]) {
  // CHECK-SAME:    -> tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT_OUT2:%.+]] = VPU.Concat([[CONV_OUT4:%.+]], [[CONV_OUT5:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE3:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE4:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE3:%.+]], [[INPUT_SLICE4:%.+]]) {
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE5:%.+]] = VPU.Slice [[CONCAT_OUT0:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE6:%.+]] = VPU.Slice [[CONCAT_OUT1:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE5:%.+]], [[INPUT_SLICE6:%.+]]) {
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT3:%.+]] = VPU.Concat([[ADD_OUT0:%.+]], [[ADD_OUT1:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE7:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE8:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 0, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT2:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE7:%.+]], [[INPUT_SLICE8:%.+]]) {
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[INPUT_SLICE9:%.+]] = VPU.Slice [[CONCAT_OUT3:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_SLICE10:%.+]] = VPU.Slice [[CONCAT_OUT2:%.+]] [0, 4608, 0, 0] [1, 4608, 4, 1] : tensor<1x9216x4x1xf16, {order = #NHWC}> to tensor<1x4608x4x1xf16, {order = #NHWC}>
  // CHECK:       [[ADD_OUT3:%.+]] = VPU.NCE.Eltwise([[INPUT_SLICE9:%.+]], [[INPUT_SLICE10:%.+]]) {
  // CHECK-SAME:  } -> tensor<1x4608x4x1xf16, {order = #NHWC}>

  // CHECK:       [[CONCAT_OUT4:%.+]] = VPU.Concat([[ADD_OUT2:%.+]], [[ADD_OUT3:%.+]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4608, 0, 0]]} : tensor<1x4608x4x1xf16, {order = #NHWC}>, tensor<1x4608x4x1xf16, {order = #NHWC}> -> tensor<1x9216x4x1xf16, {order = #NHWC}>

  // CHECK:       return [[CONCAT_OUT4:%.+]] : tensor<1x9216x4x1xf16, {order = #NHWC}>
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
    ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -1147483648 : i64, clamp_high = 1147483647 : i64,
                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    rawFilterShape = [512, 16640, 1, 1],
    strides = [1, 1]
  } : tensor<1x16640x4x1xf16, {order = #NHWC}>, tensor<512x16640x1x1xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x4x1xf16, {order = #NCHW}>

  return %0 : tensor<1x512x4x1xf16, {order = #NCHW}>


  // CHECK-DAG:  [[FILTER0:%.+]] = const.Declare tensor<512x5536x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 11104, 0, 0], [512, 5536, 1, 1]>]
  // CHECK-DAG:  [[FILTER1:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 5552, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[FILTER2:%.+]] = const.Declare tensor<512x5552x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x16640x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [512, 5552, 1, 1]>]
  // CHECK-DAG:  [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<512x1x1x4xsi32>
  // CHECK-DAG:  [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<512x1x1x4xsi32>

  // CHECK:      [[INPUT_SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT0:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE0:%.+]], [[FILTER2:%.+]], [[WEIGHTS_TABLE2:%.+]])
  // CHECK-SAME:        -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 5552, 0, 0] [1, 5552, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5552x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT1:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE1:%.+]], [[FILTER1:%.+]], [[WEIGHTS_TABLE1:%.+]])
  // CHECK-SAME:         -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[INPUT_SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 11104, 0, 0] [1, 5536, 4, 1] : tensor<1x16640x4x1xf16, {order = #NHWC}> to tensor<1x5536x4x1xf16, {order = #NHWC}>
  // CHECK:      [[CONV_OUT2:%.+]] = VPU.NCE.Convolution([[INPUT_SLICE2:%.+]], [[FILTER0:%.+]], [[WEIGHTS_TABLE0:%.+]])
  // CHECK-SAME:         -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT0:%.+]] = VPU.NCE.Eltwise([[CONV_OUT0:%.+]], [[CONV_OUT1:%.+]]) {
  // CHECK-SAME:     op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:        -> tensor<1x512x4x1xf16, {order = #NHWC}>
  // CHECK:      [[ADD_OUT1:%.+]] = VPU.NCE.Eltwise([[ADD_OUT0:%.+]], [[CONV_OUT2:%.+]]) {
  // CHECK-SAME:     op_type = #VPU.eltwise_type<ADD>
  // CHECK-SAME:        -> tensor<1x512x4x1xf16, {order = #NCHW}>
  // CHECK:      return [[ADD_OUT1:%.+]] : tensor<1x512x4x1xf16, {order = #NCHW}>
}
