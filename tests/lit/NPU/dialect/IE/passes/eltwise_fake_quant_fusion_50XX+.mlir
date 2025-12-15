//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --eltwise-fake-quantize-fusion %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
    %0 = IE.Divide(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    %1 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1xf32> -> tensor<1x12x512x1xf32>
    %3 = IE.FakeQuantize(%2, %cst_4, %cst_5, %cst_4, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    return %3 : tensor<1x12x512x1xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<11.0482254> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.97680879> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<13.025034> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       [[MUL:%.+]] = IE.Divide([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x512x1xf32>, tensor<1x12x512x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK-NOT:   IE.FakeQuantize
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[MUL]], [[CST_0]], [[CST]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x512x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x512x1xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x512x1xf32>
}
