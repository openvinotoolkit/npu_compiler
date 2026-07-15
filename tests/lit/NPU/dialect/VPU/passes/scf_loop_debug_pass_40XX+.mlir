//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --scf-loop-analysis-and-debug %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 160)>
#map2 = affine_map<(d0) -> (0, d0 - 1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 0)>
#map4 = affine_map<()[s0] -> (1, s0)>
#map5 = affine_map<(d0, d1) -> (0, d0 + d1 - 1022)>
#map6 = affine_map<(d0, d1, d2) -> (d0 - d1 - d2 + 2)>
#map7 = affine_map<(d0, d1) -> (0, d0 + d1 - 638)>

// CHECK-LABEL:   @ApplyTilingNCEConvDyn2D
module {
  net.NetworkInfo entryPoint : @ApplyTilingNCEConvDyn2D inputsInfo : {
    DataInfo "input" : tensor<1x32x?x?xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x256x?x?xf16>
  }
  func.func nested @ApplyTilingNCEConvDyn2D_func0(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
    return %0 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
  }
  func.func @ApplyTilingNCEConvDyn2D(%arg0: tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> {
    %c160 = arith.constant 160 : index
    %c256 = arith.constant 256 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg1 = %c0 to %dim step %c256 iter_args(%arg2 = %0) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = scf.for %arg3 = %c0 to %dim_0 step %c160 iter_args(%arg4 = %arg2) -> (tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>) {
        %3 = affine.min #map(%arg1)[%dim]
        %4 = affine.min #map1(%arg3)[%dim_0]
        %5 = affine.max #map2(%arg1)
        %6 = affine.max #map3(%arg1)
        %7 = affine.min #map4()[%6]
        %8 = affine.max #map5(%3, %5)
        %9 = affine.min #map4()[%8]
        %10 = affine.apply #map6(%3, %7, %9)
        %11 = affine.max #map2(%arg3)
        %12 = affine.max #map3(%arg3)
        %13 = affine.min #map4()[%12]
        %14 = affine.max #map7(%4, %11)
        %15 = affine.min #map4()[%14]
        %16 = affine.apply #map6(%4, %13, %15)
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %5, %11] [1, 32, %10, %16] [1, 1, 1, 1] : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1024, 640]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        %17 = func.call @ApplyTilingNCEConvDyn2D_func0(%extracted_slice) : (tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 256, 160]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}>
        %inserted_slice = tensor.insert_slice %17 into %arg4[0, 0, %arg1, %arg3] [1, 256, %3, %4] [1, 1, 1, 1] : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 256, 160]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
        scf.yield %inserted_slice : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
      }
      scf.yield %2 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x256x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 1024, 640]> : tensor<4xsi64>, order = #NHWC}>
  }

  // CHECK: tensor.extract_slice{{.*}}{dynamicDimsEvaluatedSizes = {H = array<i64: 257, 258>, W = array<i64: 161, 162>}}
  // CHECK: tensor.insert_slice{{.*}}{dynamicDimsEvaluatedSizes = {H = array<i64: 256>, W = array<i64: 160>}}
  // CHECK: } {uniqueStaticBlocks = 4 : i64}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 960, 26)>
#map1 = affine_map<(d0) -> (0, d0 - 1)>
#map2 = affine_map<(d0) -> (-d0 + 1, 0)>
#map3 = affine_map<()[s0] -> (1, s0)>
#map4 = affine_map<(d0, d1) -> (0, d0 + d1 - 958)>
#map5 = affine_map<(d0, d1, d2, d3) -> (0, d0 + d1 - d2 - d3 - 956)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 - d1 - d2 + d3 - d4 - d5 - 954)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (0, d0 - d1 - d2 - d3 - d4 + d5 - d6 - d7 - 952)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 + d7 - d8 - d9 - 950)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (0, d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 + d9 - d10 - d11 - 948)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (-d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 + d10 - d11 - d12 + 12)>

// CHECK-LABEL:   @ConvWithPadding
module {
  net.NetworkInfo entryPoint : @ConvWithPadding inputsInfo : {
    DataInfo "input1" : tensor<1x256x540x120xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x128x540x240xf16>
  }
  func.func @ConvWithPadding(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}> {
    %cst = arith.constant 0.000000e+00 : f16
    %c26 = arith.constant 26 : index
    %c960 = arith.constant 960 : index
    %c0 = arith.constant 0 : index
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = tensor.empty() : tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = scf.for %arg1 = %c0 to %c960 step %c26 iter_args(%arg2 = %1) -> (tensor<1x32x540x960xf16, {order = #NHWC}>) {
      %4 = affine.min #map(%arg1)
      %5 = affine.max #map1(%arg1)
      %6 = affine.max #map2(%arg1)
      %7 = affine.min #map3()[%6]
      %8 = affine.max #map4(%4, %5)
      %9 = affine.min #map3()[%8]
      %10 = affine.max #map1(%5)
      %11 = affine.max #map2(%5)
      %12 = affine.min #map3()[%11]
      %13 = affine.max #map5(%10, %4, %7, %9)
      %14 = affine.min #map3()[%13]
      %15 = affine.max #map1(%10)
      %16 = affine.max #map2(%10)
      %17 = affine.min #map3()[%16]
      %18 = affine.max #map6(%15, %12, %14, %4, %7, %9) {analyze="op2"}
      %19 = affine.min #map3()[%18]
      %20 = affine.max #map1(%15)
      %21 = affine.max #map2(%15)
      %22 = affine.min #map3()[%21]
      %23 = affine.max #map7(%20, %17, %19, %12, %14, %4, %7, %9)
      %24 = affine.min #map3()[%23]
      %25 = affine.max #map1(%20)
      %26 = affine.max #map2(%20)
      %27 = affine.min #map3()[%26]
      %28 = affine.max #map8(%25, %22, %24, %17, %19, %12, %14, %4, %7, %9) {analyze="op1"}
      %29 = affine.min #map3()[%28]
      %30 = affine.max #map1(%25)
      %31 = affine.max #map2(%25)
      %32 = affine.min #map3()[%31]
      %33 = affine.max #map9(%30, %27, %29, %22, %24, %17, %19, %12, %14, %4, %7, %9)
      %34 = affine.min #map3()[%33]
      %35 = affine.apply #map10(%32, %34, %27, %29, %22, %24, %17, %19, %12, %14, %4, %7, %9)
      %extracted_slice = tensor.extract_slice %0[0, 0, 0, %30] [1, 32, 540, %35] [1, 1, 1, 1] : tensor<1x32x540x960xf16, {order = #NHWC}> to tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded = tensor.pad %extracted_slice low[0, 0, 1, %32] high[0, 0, 1, %34] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %36 = VPU.NCE.Convolution(%padded, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_2 = tensor.pad %36 low[0, 0, 1, %27] high[0, 0, 1, %29] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %37 = VPU.NCE.Convolution(%padded_2, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_3 = tensor.pad %37 low[0, 0, 1, %22] high[0, 0, 1, %24] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %38 = VPU.NCE.Convolution(%padded_3, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %39 = VPU.NCE.DepthConvolution(%38, %cst_1) rawFilterShape [32, 1, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_4 = tensor.pad %39 low[0, 0, 1, %17] high[0, 0, 1, %19] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %40 = VPU.NCE.Convolution(%padded_4, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_5 = tensor.pad %40 low[0, 0, 1, %12] high[0, 0, 1, %14] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %41 = VPU.NCE.Convolution(%padded_5, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %padded_6 = tensor.pad %41 low[0, 0, 1, %7] high[0, 0, 1, %9] {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
        tensor.yield %cst : f16
      } : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>
      %42 = VPU.NCE.Convolution(%padded_6, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} : tensor<1x32x542x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 542, 962]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %43 = VPU.NCE.DepthConvolution(%42, %cst_1) rawFilterShape [32, 1, 1, 1] {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]} -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %43 into %arg2[0, 0, 0, %arg1] [1, 32, 540, %4] [1, 1, 1, 1] : tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 960]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x32x540x960xf16, {order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x32x540x960xf16, {order = #NHWC}>
    }
    %3 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%2 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
    return %3 : tensor<1x128x540x240xf16, {order = #NHWC}>
  }

  // CHECK: tensor.extract_slice{{.*}}{dynamicDimsEvaluatedSizes = {W = array<i64: 32, 38, 30>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.pad
  // CHECK: {dynamicDimsEvaluatedSizes = {high_pad_W = array<i64: 0, 1>, low_pad_W = array<i64: 1, 0>}}
  // CHECK: tensor.insert_slice{{.*}}{dynamicDimsEvaluatedSizes = {W = array<i64: 26, 24>}}
  // CHECK: } {uniqueStaticBlocks = 3 : i64}
}

// -----

module @NPUModule {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output/sink_port_0" tensorNames = ["output"] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  }
  func.func @main(%arg0: tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> {
    %c3 = arith.constant 3 : index
    %c276 = arith.constant 276 : index
    %c90 = arith.constant 90 : index
    %c480 = arith.constant 480 : index
    %c47 = arith.constant 47 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c2 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %0 = tensor.empty(%dim, %dim_0) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %1 = scf.for %arg1 = %c0 to %dim step %c47 iter_args(%arg2 = %0) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) {
      %8 = scf.for %arg3 = %c0 to %dim_0 step %c480 iter_args(%arg4 = %arg2) -> (tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>) {
        %9 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 47)>(%arg1)[%dim]
        %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 480)>(%arg3)[%dim_0]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, %arg3] [1, 3, %9, %10] [1, 1, 1, 1] : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> to tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 47, 480]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
        %11 = tensor.empty(%9, %10) : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 480]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        %inserted_slice = tensor.insert_slice %11 into %arg4[0, 0, %arg1, %arg3] [1, 16, %9, %10] [1, 1, 1, 1] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 47, 480]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> into tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        scf.yield %inserted_slice : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
      }
      scf.yield %8 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    }
    %dim_1 = tensor.dim %1, %c2 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %2 = arith.divsi %dim_1, %c2 : index
    %3 = arith.muli %2, %c2 : index
    %dim_2 = tensor.dim %1, %c3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %4 = arith.divsi %dim_2, %c2 : index
    %5 = arith.muli %4, %c2 : index
    %6 = tensor.empty(%3, %5) : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    %7 = scf.for %arg1 = %c0 to %3 step %c90 iter_args(%arg2 = %6) -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) {
      %8 = scf.for %arg3 = %c0 to %5 step %c276 iter_args(%arg4 = %arg2) -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>) {
        %9 = affine.max affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>(%arg3)
        %10 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%9)
        %11 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%10)
        %12 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%11)
        %13 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%12)
        %14 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%13)
        %15 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%14)
        %16 = affine.max affine_map<(d0) -> (d0 * -2 + 1, 0)>(%15)
        %17 = affine.min affine_map<()[s0] -> (1, s0)>()[%16]
        %18 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 276)>(%arg3)[%5]
        %19 = affine.max affine_map<(d0, d1)[s0] -> (d0 - s0 + d1 floordiv 2 + 2, 0)>(%9, %18)[%4]
        %20 = affine.min affine_map<()[s0] -> (1, s0)>()[%19]
        %21 = affine.max affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>(%arg3)
        %22 = affine.min affine_map<()[s0] -> (1, s0)>()[%21]
        %23 = affine.max affine_map<(d0, d1, d2, d3)[s0] -> (d0 - d1 - d2 - s0 + d3 floordiv 2 + 4, 0)>(%10, %22, %20, %18)[%4]
        %24 = affine.min affine_map<()[s0] -> (1, s0)>()[%23]
        %25 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%9)
        %26 = affine.min affine_map<()[s0] -> (1, s0)>()[%25]
        %27 = affine.max affine_map<(d0, d1, d2, d3, d4, d5)[s0] -> (d0 - d1 - d2 - d3 - d4 - s0 + d5 floordiv 2 + 6, 0)>(%11, %26, %24, %22, %20, %18)[%4]
        %28 = affine.min affine_map<()[s0] -> (1, s0)>()[%27]
        %29 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%10)
        %30 = affine.min affine_map<()[s0] -> (1, s0)>()[%29]
        %31 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - s0 + d7 floordiv 2 + 8, 0)>(%12, %30, %28, %26, %24, %22, %20, %18)[%4]
        %32 = affine.min affine_map<()[s0] -> (1, s0)>()[%31]
        %33 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%11)
        %34 = affine.min affine_map<()[s0] -> (1, s0)>()[%33]
        %35 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - s0 + d9 floordiv 2 + 10, 0)>(%13, %34, %32, %30, %28, %26, %24, %22, %20, %18)[%4]
        %36 = affine.min affine_map<()[s0] -> (1, s0)>()[%35]
        %37 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%12)
        %38 = affine.min affine_map<()[s0] -> (1, s0)>()[%37]
        %39 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - s0 + d11 floordiv 2 + 12, 0)>(%14, %38, %36, %34, %32, %30, %28, %26, %24, %22, %20, %18)[%4]
        %40 = affine.min affine_map<()[s0] -> (1, s0)>()[%39]
        %41 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%13)
        %42 = affine.min affine_map<()[s0] -> (1, s0)>()[%41]
        %43 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - d11 - d12 - s0 + d13 floordiv 2 + 14, 0)>(%15, %42, %40, %38, %36, %34, %32, %30, %28, %26, %24, %22, %20, %18)[%4]
        %44 = affine.min affine_map<()[s0] -> (1, s0)>()[%43]
        %45 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%14)
        %46 = affine.min affine_map<()[s0] -> (1, s0)>()[%45]
        %47 = affine.max affine_map<(d0) -> (0, d0 * 2 - 1)>(%15)
        %48 = affine.max affine_map<(d0) -> (d0 floordiv 2 - 1, 0)>(%arg1) {analyze="op1"}
        %49 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%48)
        %50 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%49)
        %51 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%50)
        %52 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%51)
        %53 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%52)
        %54 = affine.max affine_map<(d0) -> (0, d0 - 1)>(%53)
        %55 = affine.max affine_map<(d0) -> (d0 * -2 + 1, 0)>(%54)
        %56 = affine.min affine_map<()[s0] -> (1, s0)>()[%55]
        %57 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 90)>(%arg1)[%3]
        %58 = affine.max affine_map<(d0, d1)[s0] -> (d0 - s0 + d1 floordiv 2 + 2, 0)>(%48, %57)[%2]
        %59 = affine.min affine_map<()[s0] -> (1, s0)>()[%58]
        %60 = affine.max affine_map<(d0) -> (-(d0 floordiv 2) + 1, 0)>(%arg1)
        %61 = affine.min affine_map<()[s0] -> (1, s0)>()[%60]
        %62 = affine.max affine_map<(d0, d1, d2, d3)[s0] -> (d0 - d1 - d2 - s0 + d3 floordiv 2 + 4, 0)>(%49, %61, %59, %57)[%2]
        %63 = affine.min affine_map<()[s0] -> (1, s0)>()[%62]
        %64 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%48)
        %65 = affine.min affine_map<()[s0] -> (1, s0)>()[%64]
        %66 = affine.max affine_map<(d0, d1, d2, d3, d4, d5)[s0] -> (d0 - d1 - d2 - d3 - d4 - s0 + d5 floordiv 2 + 6, 0)>(%50, %65, %63, %61, %59, %57)[%2]
        %67 = affine.min affine_map<()[s0] -> (1, s0)>()[%66]
        %68 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%49)
        %69 = affine.min affine_map<()[s0] -> (1, s0)>()[%68]
        %70 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - s0 + d7 floordiv 2 + 8, 0)>(%51, %69, %67, %65, %63, %61, %59, %57)[%2]
        %71 = affine.min affine_map<()[s0] -> (1, s0)>()[%70]
        %72 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%50)
        %73 = affine.min affine_map<()[s0] -> (1, s0)>()[%72]
        %74 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - s0 + d9 floordiv 2 + 10, 0)>(%52, %73, %71, %69, %67, %65, %63, %61, %59, %57)[%2]
        %75 = affine.min affine_map<()[s0] -> (1, s0)>()[%74]
        %76 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%51)
        %77 = affine.min affine_map<()[s0] -> (1, s0)>()[%76]
        %78 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - s0 + d11 floordiv 2 + 12, 0)>(%53, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57)[%2]
        %79 = affine.min affine_map<()[s0] -> (1, s0)>()[%78]
        %80 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%52)
        %81 = affine.min affine_map<()[s0] -> (1, s0)>()[%80]
        %82 = affine.max affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13)[s0] -> (d0 - d1 - d2 - d3 - d4 - d5 - d6 - d7 - d8 - d9 - d10 - d11 - d12 - s0 + d13 floordiv 2 + 14, 0)>(%54, %81, %79, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57)[%2]
        %83 = affine.min affine_map<()[s0] -> (1, s0)>()[%82]
        %84 = affine.max affine_map<(d0) -> (-d0 + 1, 0)>(%53)
        %85 = affine.min affine_map<()[s0] -> (1, s0)>()[%84]
        %86 = affine.max affine_map<(d0) -> (0, d0 * 2 - 1)>(%54)
        %87 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%9, %47)
        %88 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%48, %86) {analyze="op2"}
        %89 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15) -> (-d0 - d1 * 2 - d2 * 2 - d3 * 2 - d4 * 2 - d5 * 2 - d6 * 2 - d7 * 2 - d8 * 2 - d9 * 2 - d10 * 2 - d11 * 2 - d12 * 2 - d13 * 2 - d14 * 2 + (d15 floordiv 2) * 2 + 29)>(%17, %46, %44, %42, %40, %38, %36, %34, %32, %30, %28, %26, %24, %22, %20, %18)
        %90 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15) -> (-d0 - d1 * 2 - d2 * 2 - d3 * 2 - d4 * 2 - d5 * 2 - d6 * 2 - d7 * 2 - d8 * 2 - d9 * 2 - d10 * 2 - d11 * 2 - d12 * 2 - d13 * 2 - d14 * 2 + (d15 floordiv 2) * 2 + 29)>(%56, %85, %83, %81, %79, %77, %75, %73, %71, %69, %67, %65, %63, %61, %59, %57)
        scf.yield %arg2 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
      } {upperbound=726}
      scf.yield %8 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
    } {upperbound=450}
    return %7 : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 1080, 1920]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}>
  }

  // CHECK: affine.max{{.*}} {analyze = "op1", dynamicDimsEvaluatedSizes = {results = array<i64: 0, 44, 89, 134, 179>}}
  // CHECK: affine.apply{{.*}}{analyze = "op2", dynamicDimsEvaluatedSizes = {results = array<i64: 0, -31, -76, -121, -166>}}
}
