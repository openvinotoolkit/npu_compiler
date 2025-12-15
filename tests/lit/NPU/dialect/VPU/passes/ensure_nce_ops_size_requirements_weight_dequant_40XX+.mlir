//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements="enable-dequant-weight-ensurance-before-strategy=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.0014466386799718818:108>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

// CHECK-LABEL:   @EnableWeightDeqauntEnsuranceBeforeStrategy
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1280x256x1xf16, {order = #NHWC}>
func.func @EnableWeightDeqauntEnsuranceBeforeStrategy(%arg0: tensor<1x1280x256x1xf16, {order = #NHWC}>) -> tensor<1x10240x64x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<10240x1280x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<1280x10240xui8>, [#const.Reshape<[1, 1, 1280, 10240]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 1280, 1, 10240]>, #const.Transpose<#NWHC>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [10240, 1280, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]
    %cst_0 = const.Declare tensor<10240x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>
    %0 = VPU.Dequantize(%cst) {dstElemType = f16} : tensor<10240x1280x1x1x!qElemType, {order = #NHWC}> -> tensor<10240x1280x1x1xf16, {order = #NHWC}>
    %1 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 64, 4]} : tensor<1x1280x256x1xf16, {order = #NHWC}> -> tensor<1x1280x64x4xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %0, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [10240, 1280, 1, 1], strides = [1, 1]} : tensor<1x1280x64x4xf16, {order = #NHWC}>, tensor<10240x1280x1x1xf16, {order = #NHWC}>, tensor<10240x1x1x4xsi32> -> tensor<1x10240x64x4xf16, {order = #NHWC}>
   return %2 : tensor<1x10240x64x4xf16, {order = #NHWC}>

  // CHECK: [[CST:%.+]] = const.Declare tensor<10240x1280x1x1x!qElemType, {order = #NHWC}> = dense<10> :
  // CHECK-SAME{LITERAL}: tensor<1280x10240xui8>, [#const.Reshape<[1, 1, 1280, 10240]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 1280, 1, 10240]>, #const.Transpose<#NWHC>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [10240, 1280, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]

  // CHECK: [[CST0:%.+]] = const.Declare tensor<10240x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>

  // CHECK: [[WT_DEQUANT:%.+]] = VPU.Dequantize([[CST]]) {dstElemType = f16} : tensor<10240x1280x1x1x!qElemType, {order = #NHWC}> -> tensor<10240x1280x1x1xf16, {order = #NHWC}>

  // CHECK: [[ACTIVATION:%.+]] = VPU.AffineReshape([[INPUT]])
  // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 1280, 64, 4]} : tensor<1x1280x256x1xf16, {order = #NHWC}> -> tensor<1x1280x64x4xf16, {order = #NHWC}>

  // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[ACTIVATION]], [[WT_DEQUANT]], [[CST0]]) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
  // CHECK-SAME: pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME: ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
  // CHECK-SAME: rawFilterShape = [10240, 1280, 1, 1], strides = [1, 1]}
  // CHECK-SAME: tensor<1x1280x64x4xf16, {order = #NHWC}>, tensor<10240x1280x1x1xf16, {order = #NHWC}>, tensor<10240x1x1x4xsi32> -> tensor<1x10240x64x4xf16, {order = #NHWC}>

  // CHECK: return [[CONV]] : tensor<1x10240x64x4xf16, {order = #NHWC}>

}
