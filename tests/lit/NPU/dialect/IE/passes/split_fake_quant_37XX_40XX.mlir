//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --split-fake-quant %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

!qElemType = !quant.uniform<i8<-127:127>:f16:0, {0.011811023622047244:-42,0.0090543491633858271:-6,0.010630690206692913:-14}>

// CHECK-LABEL: @ConstantsDequantizeSplitFakeQuantForMultiZP
func.func @ConstantsDequantizeSplitFakeQuantForMultiZP(%arg0: tensor<1x3x16x16xf16>) -> tensor<1x3x16x16xf16> {
    %cst = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>]
    %cst_low = const.Declare tensor<3x1x1x1xf16> = dense<[[[[-1.0]]], [[[-1.1]]], [[[-1.2]]]]> : tensor<3x1x1x1xf16>
    %cst_high = const.Declare tensor<3x1x1x1xf16> = dense<[[[[2.0]]], [[[1.2]]], [[[1.5]]]]> : tensor<3x1x1x1xf16>

    %0 = IE.FakeQuantize(%cst, %cst_low, %cst_high, %cst_low, %cst_high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<3x3x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16>, tensor<3x1x1x1xf16> -> tensor<3x3x1x1xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    return %1 : tensor<1x3x16x16xf16>

    // CHECK-NOT:   IE.FakeQuantize
    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16> = dense<9> : tensor<3x3x1x1xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.Dequantize]
    // CHECK:       [[CONV:%.+]] =  IE.Convolution(%arg0, [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x16x16xf16>, tensor<3x3x1x1xf16> -> tensor<1x3x16x16xf16>

    // CHECK:       return [[CONV]] : tensor<1x3x16x16xf16>
}
