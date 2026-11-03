//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --eltwise-fake-quantize-fusion %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

// CHECK-LABEL: @AddFakeQuantizeFusionPerTensorFQRhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x512x1xf32>
func.func @AddFakeQuantizeFusionPerTensorFQRhs(%arg0: tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32> {
    %cst = const.Declare tensor<1xf32> = dense<2.510000e+02> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %cst_1 = const.Declare tensor<1xf32> = dense<2.550000e+02> : tensor<1xf32>
    %cst_2 = const.Declare tensor<1xf32> = dense<-2.05716681> : tensor<1xf32>
    %cst_3 = const.Declare tensor<1xf32> = dense<2.04109526> : tensor<1xf32>
    %cst_4 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_5 = const.Declare tensor<1x1x1x1xf32> = dense<13.025034> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1xf32> -> tensor<1x12x512x1xf32>
    %3 = IE.FakeQuantize(%2, %cst_4, %cst_5, %cst_4, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    return %3 : tensor<1x12x512x1xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<11.0482254> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.97680879> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<13.025034> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:       [[MUL:%.+]] = IE.Multiply([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK-NOT:   IE.FakeQuantize
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[MUL]], [[CST_0]], [[CST]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x512x1xf32>
}

// -----

!qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @AddFakeQuantizeFusionPerTensorDQRhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x512x1xf32>
func.func @AddFakeQuantizeFusionPerTensorDQRhs(%arg0: tensor<1x12x512x1xf32>) -> tensor<1x12x512x1xf32> {
    %cst = const.Declare tensor<1x!qElemType> = dense<3.0> : tensor<1xf32>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>]
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<13.025034> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    %1 = IE.Dequantize(%cst) {dstElemType = f32} : tensor<1x!qElemType> -> tensor<1xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1xf32> -> tensor<1x12x512x1xf32>
    %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    return %3 : tensor<1x12x512x1xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<13.025034> : tensor<1x1x1x1xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-7.38823509> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.63679886> : tensor<1x1x1x1xf32>
    // CHECK-NOT:   IE.Dequantize
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[MUL]], [[CST_1]], [[CST_2]], [[CST]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x512x1xf32>
}
