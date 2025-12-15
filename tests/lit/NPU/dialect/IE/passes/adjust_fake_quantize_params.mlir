//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-fake-quantize-params %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX
// CHECK-LABEL: @AdjustFakeQuantizeWithFQ

// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQuantizeWithFQ(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x1x1xf32> {
    %fq1_in_low = const.Declare tensor<1x1x1x1xf32> = dense <-182.1645050048828> : tensor<1x1x1x1xf32> isSplat
    %fq1_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <392.4757080078125> : tensor<1x1x1x1xf32> isSplat
    %fq1_out_low = const.Declare tensor<1x1x1x1xf32> = dense <-182.1645050048828> : tensor<1x1x1x1xf32> isSplat
    %fq1_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <392.4757080078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq3_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq3_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <504.5491638183594> : tensor<1x1x1x1xf32> isSplat
    %fq3_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq3_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <504.5491638183594> : tensor<1x1x1x1xf32> isSplat
    %add_cst = const.Declare tensor<1x1x1x1xf32> = dense <8.537536814401392e-06> : tensor<1x1x1x1xf32> isSplat

    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq1_in_low, %fq1_in_hi, %fq1_out_low, %fq1_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.ReduceMean(%3) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    %5 = IE.FakeQuantize(%4, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    %6 = IE.Add(%5, %add_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>

    return %6 : tensor<1x128x1x1xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<50.4549179> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<16030.208> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<124.111717> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-57.6054726> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<392.475708> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-182.164505> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_7:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ_0:%.+]] = IE.FakeQuantize([[SUB]], [[CST_6]], [[CST_5]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL:%.+]] = IE.Multiply([[FQ_0]], [[FQ_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ_1:%.+]] = IE.FakeQuantize([[MUL]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[RM:%.+]] = IE.ReduceMean([[FQ_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[FQ_2:%.+]] = IE.FakeQuantize([[RM]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[ADD:%.+]] = IE.Add([[FQ_2]], [[CST_7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
}

// -----

// CHECK-LABEL: @AdjustFakeQuantizeWithMultipleUsers
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQuantizeWithMultipleUsers(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> (tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>) {
    %fq1_in_low = const.Declare tensor<1x1x1x1xf32> = dense <-182.1645050048828> : tensor<1x1x1x1xf32> isSplat
    %fq1_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <392.4757080078125> : tensor<1x1x1x1xf32> isSplat
    %fq1_out_low = const.Declare tensor<1x1x1x1xf32> = dense <-182.1645050048828> : tensor<1x1x1x1xf32> isSplat
    %fq1_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <392.4757080078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq3_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq3_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <504.5491638183594> : tensor<1x1x1x1xf32> isSplat
    %fq3_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq3_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <504.5491638183594> : tensor<1x1x1x1xf32> isSplat
    %mul_in = const.Declare tensor<1x1x1x1xf32> = dense <8.537536814401392e-06> : tensor<1x1x1x1xf32> isSplat

    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq1_in_low, %fq1_in_hi, %fq1_out_low, %fq1_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.ReduceMean(%3) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    %5 = IE.FakeQuantize(%4, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    %6 = IE.Multiply(%3, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %7 = IE.FakeQuantize(%6, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>

    return %5, %7 : tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<50.4549179> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<16030.208> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<124.111717> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-57.6054726> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<392.475708> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-182.164505> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_7:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>

    // CHECK: [[SUB:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ_0:%.+]] = IE.FakeQuantize([[SUB]], [[CST_6]], [[CST_5]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL:%.+]] = IE.Multiply([[FQ_0]], [[FQ_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ_1:%.+]] = IE.FakeQuantize([[MUL]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[RM:%.+]] = IE.ReduceMean([[FQ_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[FQ_2:%.+]] = IE.FakeQuantize([[RM]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[FQ_1]], [[CST_7]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ_3:%.+]] = IE.FakeQuantize([[MUL1]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
}
