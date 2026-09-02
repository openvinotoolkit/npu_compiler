//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --split-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


!qElemType = !quant.uniform<i8<-127:127>:f16:0, {0.011811023622047244:-42,0.0090543491633858271:-6,0.010630690206692913:-14}>

// CHECK-LABEL: @ConstantsDequantizeSplitFakeQuantForMultiZP
// CHECK-SAME:      [[ARG_0:%[^:]+]]: tensor<1x3x16x16xf16>
func.func @ConstantsDequantizeSplitFakeQuantForMultiZP(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %cst = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %cst_low = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]]]> : tensor<3x1x1x1xf16>
    %cst_high = const.Declare tensor<3x1x1x1xf16> = dense<[[[[2.0]]], [[[1.2]]], [[[1.5]]]]> : tensor<3x1x1x1xf16>

    %0 = IE.FakeQuantize(%cst, %cst_low, %cst_high, %cst_low, %cst_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<3x3x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK-NOT:   IE.FakeQuantize
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Dequantize]
    // CHECK:       [[CONV:%.+]] =  IE.Convolution([[ARG_0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    // CHECK:       return [[CONV]] : tensor<1x3x16x16xf16>
}

// -----

// CHECK: !qElemType = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK: !qElemType1 = !quant.uniform<i4:f16, 6.250000e-02>
!qI4Sym = !quant.uniform<i4:f16, 0.0625>

// CHECK-LABEL: @SplitFakeQuantActSignedAlignedToSymSI4Weights
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x16x16xf16>
func.func @SplitFakeQuantActSignedAlignedToSymSI4Weights(%arg0: tensor<1x32x16x16xf16>) -> tensor<1x64x14x14xf16> {
    %lo = const.Declare tensor<1x1x1x1xf16> = dense<-1.280000e+02> : tensor<1x1x1x1xf16>
    %hi = const.Declare tensor<1x1x1x1xf16> = dense<1.270000e+02> : tensor<1x1x1x1xf16>
    %fq = IE.FakeQuantize(%arg0, %lo, %hi, %lo, %hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x16x16xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x16x16xf16>
    %w = const.Declare tensor<64x32x3x3x!qI4Sym> = dense<1> : tensor<64x32x3x3xsi4>, [#const.CastElemType<si4>, #const.CastElemType<!qI4Sym>]
    %wdq = IE.Dequantize(%w) {dstElemType = f16} : tensor<64x32x3x3x!qI4Sym> -> tensor<64x32x3x3xf16>
    %conv = IE.Convolution(%fq, %wdq) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x16x16xf16>, tensor<64x32x3x3xf16> -> tensor<1x64x14x14xf16>
    return %conv : tensor<1x64x14x14xf16>

    // Activation is quantized to a SIGNED i8 type, matching the signed i4 weights that stay signed on arch50xx.
    // CHECK:      [[ACT:%.+]] = IE.Quantize([[ARG0]]) {dstElemType = !qElemType} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16x!qElemType>
    // CHECK:      [[ACT_DQ:%.+]] = IE.Dequantize([[ACT]]) {dstElemType = f16} : tensor<1x32x16x16x!qElemType> -> tensor<1x32x16x16xf16>
    // CHECK:      [[W:%.+]] = const.Declare tensor<64x32x3x3x!qElemType1> = dense<1> : tensor<64x32x3x3xsi4>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType1>]
    // CHECK:      [[W_DQ:%.+]] = IE.Dequantize([[W]]) {dstElemType = f16} : tensor<64x32x3x3x!qElemType1> -> tensor<64x32x3x3xf16>
    // CHECK:      [[CONV:%.+]] = IE.Convolution([[ACT_DQ]], [[W_DQ]])
    // CHECK:      return [[CONV]] : tensor<1x64x14x14xf16>
}
