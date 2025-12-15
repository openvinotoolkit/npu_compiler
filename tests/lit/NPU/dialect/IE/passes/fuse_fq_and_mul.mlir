//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-fq-and-mul="fuse-fq-and-mul-with-non-const-input=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseFQAndMulAtConstWeightsAndMulLhsIsActivation
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseFQAndMulAtConstWeightsAndMulLhsIsActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst_0 = const.Declare tensor<288x16x3x3xf32> = dense<1.0> : tensor<288x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>
    %cst_4 = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_5 = const.Declare tensor<288x1x1x1xf32> = dense<2.0> : tensor<288x1x1x1xf32>
    %1 = IE.Multiply(%0, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<288x16x3x3xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_6 = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    %2 = IE.Reshape(%1, %cst_6) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    %3 = IE.GroupConvolution(%arg0, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %3 : tensor<1x288x20x20xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<288x16x3x3xf32> = dense<1.000000e+00> : tensor<288x16x3x3xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST5:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CST1]], [[CST2]], [[CST3]], [[CST4]], [[CST5]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[FQ]], [[CST0]]) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    // CHECK:       [[CONV:%.+]]  = IE.GroupConvolution([[INPUT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @FuseFQAndMulAtConstWeightsAndMulRhsIsActivation
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseFQAndMulAtConstWeightsAndMulRhsIsActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst_0 = const.Declare tensor<288x16x3x3xf32> = dense<1.0> : tensor<288x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>
    %cst_4 = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    %cst_5 = const.Declare tensor<288x1x1x1xf32> = dense<2.0> : tensor<288x1x1x1xf32>
    %1 = IE.Multiply(%cst_5, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<288x1x1x1xf32>, tensor<288x16x3x3xf32> -> tensor<288x16x3x3xf32>
    %cst_6 = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    %2 = IE.Reshape(%1, %cst_6) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    %3 = IE.GroupConvolution(%arg0, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %3 : tensor<1x288x20x20xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<5xsi64> = dense<[18, 16, 16, 3, 3]> : tensor<5xsi64>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<288x16x3x3xf32> = dense<1.000000e+00> : tensor<288x16x3x3xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<-1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST5:%.+]] = const.Declare tensor<288x1x1x1xf32> = dense<1.270000e+02> : tensor<288x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CST1]], [[CST2]], [[CST3]], [[CST4]], [[CST5]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<288x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<288x1x1x1xf32>, tensor<288x1x1x1xf32> -> tensor<288x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[FQ]], [[CST0]]) : tensor<288x16x3x3xf32>, tensor<5xsi64> -> tensor<18x16x16x3x3xf32>
    // CHECK:       [[CONV:%.+]]  = IE.GroupConvolution([[INPUT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @FuseFQAndMulAtActivation
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x288x20x20xf32>
func.func @FuseFQAndMulAtActivation(%arg0: tensor<1x288x20x20xf32>) -> tensor<1x288x20x20xf32> {
    %cst = const.Declare tensor<1x288x1x1xf32> = dense<2.000000e+00> : tensor<1x288x1x1xf32>
    %cst_0 = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x288x20x20xf32>, tensor<1x288x1x1xf32> -> tensor<1x288x20x20xf32>
    %2 = IE.GroupConvolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    return %2 : tensor<1x288x20x20xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<18x16x16x3x3xf32> = dense<1.000000e+00> : tensor<18x16x16x3x3xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x288x20x20xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x288x20x20xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[CONV:%.+]]  = IE.GroupConvolution([[FQ]], [[CST0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x288x20x20xf32>, tensor<18x16x16x3x3xf32> -> tensor<1x288x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x288x20x20xf32>
}

// -----

// CHECK-LABEL: @FuseFQAndMulAtConstWeightsAndMul1DConst
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x128x20x20xf32>
func.func @FuseFQAndMulAtConstWeightsAndMul1DConst(%arg0: tensor<1x128x20x20xf32>) -> tensor<1x128x20x20xf32> {
    %cst_0 = const.Declare tensor<128x16x3x3xf32> = dense<1.0> : tensor<128x16x3x3xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<128x1x1x1xf32> = dense<-1.270000e+02> : tensor<128x1x1x1xf32>
    %cst_4 = const.Declare tensor<128x1x1x1xf32> = dense<1.270000e+02> : tensor<128x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst_0, %cst_1, %cst_2, %cst_3, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<128x1x1x1xf32> -> tensor<128x16x3x3xf32>
    %cst_5 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %1 = IE.Multiply(%0, %cst_5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x16x3x3xf32>, tensor<1xf32> -> tensor<128x16x3x3xf32>
    %cst_6 = const.Declare tensor<5xsi64> = dense<[8, 16, 16, 3, 3]> : tensor<5xsi64>
    %2 = IE.Reshape(%1, %cst_6) : tensor<128x16x3x3xf32>, tensor<5xsi64> -> tensor<8x16x16x3x3xf32>
    %3 = IE.GroupConvolution(%arg0, %2) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x20x20xf32>, tensor<8x16x16x3x3xf32> -> tensor<1x128x20x20xf32>

    return %3 : tensor<1x128x20x20xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<5xsi64> = dense<[8, 16, 16, 3, 3]> : tensor<5xsi64>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<128x16x3x3xf32> = dense<1.000000e+00> : tensor<128x16x3x3xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<128x1x1x1xf32> = dense<-1.270000e+02> : tensor<128x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST5:%.+]] = const.Declare tensor<128x1x1x1xf32> = dense<1.270000e+02> : tensor<128x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CST1]], [[CST2]], [[CST3]], [[CST4]], [[CST5]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<128x1x1x1xf32> -> tensor<128x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[FQ]], [[CST0]]) : tensor<128x16x3x3xf32>, tensor<5xsi64> -> tensor<8x16x16x3x3xf32>
    // CHECK:       [[CONV:%.+]]  = IE.GroupConvolution([[INPUT]], [[RESHAPE]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x128x20x20xf32>, tensor<8x16x16x3x3xf32> -> tensor<1x128x20x20xf32>

    // CHECK:       return [[CONV]] : tensor<1x128x20x20xf32>
}

// -----

// CHECK-LABEL: @FusePerTensorFQAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<128x16x3x3xf32>
func.func @FusePerTensorFQAndSplatMul(%arg0: tensor<128x16x3x3xf32>) -> tensor<128x16x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<128x16x3x3xf32>
    %cst_4 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x16x3x3xf32>, tensor<1xf32> -> tensor<128x16x3x3xf32>

    return %1 : tensor<128x16x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>, [#const.Rescale<2.000000e+00 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<128x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<128x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<128x16x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerChannelFQAndSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @FusePerChannelFQAndSplatMul(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x6x1x1xf32> = dense<1.100000> : tensor<1x6x1x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+00]], [[-2.000000e+00]], [[-3.000000e+00]], [[-4.000000e+00]], [[-5.000000e+00]], [[-6.000000e+00]]]]> : tensor<1x6x1x1xf32>, [#const.Rescale<1.1000000238418579 : f64>]
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]]]]> : tensor<1x6x1x1xf32>, [#const.Rescale<1.1000000238418579 : f64>]
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerTensorFQAndNoneSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x16x3x3xf32>
func.func @FusePerTensorFQAndNoneSplatMul(%arg0: tensor<1x16x3x3xf32>) -> tensor<1x16x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x16x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x1xf32> = dense<[[[[1.0], [1.1], [1.2]]]]> : tensor<1x1x3x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x16x3x3xf32>

    return %1 : tensor<1x16x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+02], [-1.100000e+02], [-120.000008]]]]> : tensor<1x1x3x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+02], [1.100000e+02], [120.000008]]]]> : tensor<1x1x3x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x16x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x3x1xf32>, tensor<1x1x3x1xf32> -> tensor<1x16x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x16x3x3xf32>
}

// -----

// CHECK-LABEL: @FusePerChannelFQAndNoneSplatMul
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @FusePerChannelFQAndNoneSplatMul(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.1]], [[2.2]], [[3.3]], [[4.4]], [[5.5]], [[6.6]]]]> : tensor<1x6x1x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.100000e+00]], [[-4.400000e+00]], [[-9.89999961]], [[-1.760000e+01]], [[-2.750000e+01]], [[-3.960000e+01]]]]> : tensor<1x6x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.100000e+00]], [[4.400000e+00]], [[9.89999961]], [[1.760000e+01]], [[2.750000e+01]], [[3.960000e+01]]]]> : tensor<1x6x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST0]], [[CST1]], [[CST2]], [[CST3]]) {
    // CHECK-SAME:                      auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
    // CHECK-SAME:                  } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK-NOT:   IE.Multiply

    // CHECK:       return [[FQ]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @NotFusePerChannelFQAndNoneSplatMulWithDiffShape
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @NotFusePerChannelFQAndNoneSplatMulWithDiffShape(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]]]> : tensor<1x6x1x1xf32>
    %cst_3 = const.Declare tensor<1x6x1x1xf32> = dense<[[[[-1.0]], [[-2.0]], [[-3.0]], [[-4.0]], [[-5.0]], [[-6.0]]]]> : tensor<1x6x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x1xf32> = dense<[[[[1.0], [1.1], [1.2]]]]> : tensor<1x1x3x1xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x3x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00], [1.100000e+00], [1.200000e+00]]]]> : tensor<1x1x3x1xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]]]]> : tensor<1x6x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x6x1x1xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[-1.000000e+00]], [[-2.000000e+00]], [[-3.000000e+00]], [[-4.000000e+00]], [[-5.000000e+00]], [[-6.000000e+00]]]]> : tensor<1x6x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x6x1x1xf32>, tensor<1x6x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x1xf32> -> tensor<1x6x3x3xf32>

    // CHECK:       return [[MUL]] : tensor<1x6x3x3xf32>
}

// -----

// CHECK-LABEL: @NotFusePerTensorFQAndNoneSplatMulWithMultiAxis
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x6x3x3xf32>
func.func @NotFusePerTensorFQAndNoneSplatMulWithMultiAxis(%arg0: tensor<1x6x3x3xf32>) -> tensor<1x6x3x3xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    %cst_3 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_2, %cst_3) {
                auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64
            } : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x6x3x3xf32>
    %cst_4 = const.Declare tensor<1x1x3x3xf32> = dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]> : tensor<1x1x3x3xf32>
    %1 = IE.Multiply(%0, %cst_4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<1x6x3x3xf32>

    return %1 : tensor<1x6x3x3xf32>

    // CHECK-DAG:   [[CST0:%.+]] = const.Declare tensor<1x1x3x3xf32> = dense<
    // CHECK-SAME{LITERAL}:         [[[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00, 9.000000e+00]]]]> : tensor<1x1x3x3xf32>
    // CHECK-DAG:   [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST1]], [[CST2]], [[CST3]], [[CST4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x6x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x6x3x3xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x6x3x3xf32>, tensor<1x1x3x3xf32> -> tensor<1x6x3x3xf32>

    // CHECK:       return [[MUL]] : tensor<1x6x3x3xf32>
}
