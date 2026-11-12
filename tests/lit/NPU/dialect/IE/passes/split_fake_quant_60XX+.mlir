//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --split-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU6010


!qElemType = !quant.uniform<i8:f16:0, {0.011764705882352941:-43,0.009018841911764705:-6,0.010589001225490197:-15}>

// Not Supported Yet, There is HW support.

// CHECK-LABEL: @ConstantsSplitFakeQuantForMultiZP
// CHECK:       [[ARG0:%.+]]: tensor<1x3x16x16xf16>
func.func @ConstantsSplitFakeQuantForMultiZP(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %cst = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %cst_low = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]]]> : tensor<3x1x1x1xf16>
    %cst_high = const.Declare tensor<3x1x1x1xf16> = dense<[[[[2.0]]], [[[1.2]]], [[[1.5]]]]> : tensor<3x1x1x1xf16>

    %0 = IE.FakeQuantize(%cst, %cst_low, %cst_high, %cst_low, %cst_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<3x3x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK:       [[CST:%.+]] = const.Declare  tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Dequantize]

    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[CST]])
    // CHECK:       return [[CONV]] : tensor<1x3x16x16xf16>
}


// -----

!qElemType = !quant.uniform<i8:f16, 0.011764705882352941:-43>

// CHECK-LABEL: @ConstantsSplitFakeQuantForPerTensorZP
// CHECK:       [[ARG0:%.+]]: tensor<1x3x16x16xf16>

func.func @ConstantsSplitFakeQuantForPerTensorZP(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %cst = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %cst_low = const.Declare tensor<1x1x1x1xf16> = dense<[[[[-1.0]]]]> : tensor<1x1x1x1xf16>
    %cst_high = const.Declare tensor<1x1x1x1xf16> = dense<[[[[2.0]]]]> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%cst, %cst_low, %cst_high, %cst_low, %cst_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<3x3x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK-NOT:   IE.FakeQuantize
    // CHECK:   [[CST:%.+]] =  const.Declare tensor<3x3x1x1xsi8> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.Quantize<!qElemType>, #const.CastElemType<si8>]
    // CHECK: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> =  dense<0.0117647061> : tensor<1x1x1x1xf32>
    // CHECK: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xsi8> = dense<-43> : tensor<1x1x1x1xsi8>
    // CHECK:    [[DQ:%.+]] = IE.DynamicDequantize([[CST]], [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.synthetic_dyn_dequant} : tensor<3x3x1x1xsi8>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xsi8> -> tensor<3x3x1x1xf16>
    // CHECK:       [[CONV:%.+]] =  IE.Convolution([[ARG0]], [[DQ]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    // CHECK:       return [[CONV]] : tensor<1x3x16x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 6.250000e-02:159>

module @TestSplitFakeQuantActUnsignedAlignedToU8Weights {

  // CHECK: !qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>
  // CHECK: !qElemType1 = !quant.uniform<u8:f16, 6.250000e-02:159>
  // CHECK-LABEL: @SplitFakeQuantActUnsignedAlignedToU8Weights
  // CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x16x16xf16>
  func.func @SplitFakeQuantActUnsignedAlignedToU8Weights(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x64x14x14xf16> {
      %lo = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
      %hi = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
      %fq = IE.FakeQuantize(%arg0, %lo, %hi, %lo, %hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x32x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x16x16xf16>
      %w = const.Declare tensor<64x32x3x3x!qElemType> = dense<1> : tensor<64x32x3x3xui8>, [ #const.CastElemType<!qElemType>]
      %wdq = IE.Dequantize(%w) {dstElemType = f16} : tensor<64x32x3x3x!qElemType> -> tensor<64x32x3x3xf16>
      %conv = IE.Convolution(%fq, %wdq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x16x16xf16>, tensor<64x32x3x3xf16> -> tensor<1x64x14x14xf16>
      return %conv : tensor<1x64x14x14xf16>

      // CHECK:      [[ACT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16x!qElemType>
      // CHECK:      [[ACT_DQ:%.+]] = IE.Dequantize([[ACT]]) {dstElemType = f16} : tensor<1x32x16x16x!qElemType> -> tensor<1x32x16x16xf16>
      // CHECK:      [[W:%.+]] = const.Declare tensor<64x32x3x3x!qElemType1> = dense<1> : tensor<64x32x3x3xui8>, [#const.CastElemType<!qElemType1>]
      // CHECK:      [[W_DQ:%.+]] = IE.Dequantize([[W]]) {dstElemType = f16} : tensor<64x32x3x3x!qElemType1> -> tensor<64x32x3x3xf16>
      // CHECK:      [[CONV:%.+]] = IE.Convolution([[ACT_DQ]], [[W_DQ]])
      // CHECK:      return [[CONV]] : tensor<1x64x14x14xf16>
  }
}
