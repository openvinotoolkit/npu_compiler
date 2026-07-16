//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-vpunn-cost-for-tiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

  // CHECK-LABEL: @ConvertPermute4Ops2DimDynamic
  // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  func.func @ConvertPermute4Ops2DimDynamic(%arg0: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %cst_1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_1) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]}: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_2) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]}: tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

	// CHECK: [[CST:%.+]] = const.Declare
	// CHECK: [[CST_0:%.+]] = const.Declare

	// CHECK:   [[CONVERT:%.+]] = VPU.Convert([[INPUT]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, {{[2-9]|[1-9][0-9]+}}]

	// CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, {{[2-9]|[1-9][0-9]+}}]

	// CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[PERMUTE]], [[CST]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, {{[2-9]|[1-9][0-9]+}}]

	// CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_1]], [[CST_0]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, {{[2-9]|[1-9][0-9]+}}]

	// CHECK: return [[CONV_2]] : tensor<1x16x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

  // CHECK-LABEL: @ConvertPermute4Ops1DimDynamic
  // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
  func.func @ConvertPermute4Ops1DimDynamic(%arg0: tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
    %cst_1 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x?x1280xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_1) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]}: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_2) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1]}: tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
    return %3 : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>

	// CHECK: [[CST:%.+]] = const.Declare
	// CHECK: [[CST_0:%.+]] = const.Declare

	// CHECK:   [[CONVERT:%.+]] = VPU.Convert([[INPUT]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, 1]

	// CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[CONVERT]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, 1]

	// CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[PERMUTE]], [[CST]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, 1]

	// CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_1]], [[CST_0]])
	// CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, 1]

	// CHECK: return [[CONV_2]] : tensor<1x16x?x1280xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

  // CHECK-LABEL: @PermuteIsolatedCheck
  // CHECK-SAME: ([[INPUT:%.+]]: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}>
func.func @PermuteIsolatedCheck(%arg0: tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NCHW}>) -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = #NHWC}> {
  %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, expandedChannels = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  return %0 : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 1280, 1280]> : tensor<4xsi64>, order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

   // CHECK: [[PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]])
   // CHECK-SAME: tilingStrategy = [1, 1, {{[2-9]|[1-9][0-9]+}}, {{[2-9]|[1-9][0-9]+}}]
}
