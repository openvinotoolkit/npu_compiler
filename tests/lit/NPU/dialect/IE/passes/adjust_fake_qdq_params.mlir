//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-fake-qdq-params %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// This is single FQ op check, important because it also checks that when we reach top of the
// graph we can insert a multiply to an input argument.
// The defining op of an input argument is nullptr and needs to be handled correctly.
// CHECK-LABEL: @AdjustSingleFakeQuantizeOp
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustSingleFakeQuantizeOp(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %1 = IE.FakeQuantize(%arg0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    return %1 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>

    // CHECK: [[MUL0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ1:%.+]] = IE.FakeQuantize([[MUL0]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[FQ1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}



// -----

// This test checks propagation of multiply operations all the way to the top and bottom.
// The traversal also encounters a subtract and an add operation.
// CHECK-LABEL: @AdjustFQBetweenSubAndAdd
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFQBetweenSubAndAdd(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Add(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>

    // CHECK-NEXT: [[MUL0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[INPUT_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[SUB3:%.+]] = IE.Subtract([[MUL2]], [[MUL1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ4:%.+]] = IE.FakeQuantize([[SUB3]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[ADD5:%.+]] = IE.Add([[FQ4]], [[MUL0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL6:%.+]] = IE.Multiply([[ADD5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}

// -----

// Test to check propagation through multiply ops with non-identical inputs both in the up and down directions.
// CHECK-LABEL: @AdjustFQMul
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFQMul(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}

// -----


// Test to check propagation of through a multiply op at the top with identical inputs
// CHECK-LABEL: @AdjustFQMulSquareUp
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulSquareUp(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}


// -----

// Test to check propagation of multiply op down with square inputs
// CHECK-LABEL: @AdjustFQMulSquareDown
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulSquareDown(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %mul_in = const.Declare tensor<1x1x1x1xf32> = dense <8.537536814401392e-06> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>, [#const.Rescale<0.20441406965255737 : f64>]
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.FakeQuantize([[OUT0]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Multiply([[OUT1]], [[OUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}

// -----

// Test to check propagation of multiply op down with square inputs and fusion of multiply factor in
// up propagation in particular propagated mul does not reach all the way to subtract.
// CHECK-LABEL: @AdjustFQMulUpFuseConst
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulUpFuseConst(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %mul_in = const.Declare tensor<1x1x1x1xf32> = dense <8.537536814401392e-06> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Subtract(%arg0, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.Multiply(%0, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.FakeQuantize(%1, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.Multiply(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %3 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>, [#const.Rescale<0.20441406965255737 : f64>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>
    // CHECK: [[OUT0:%.+]] = IE.Subtract(%arg0, [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[OUT0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>


}



// -----

// Test to check propagation of multiply ops with square inputs both up and down
// CHECK-LABEL: @AdjustFQMulSquareUpDown
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulSquareUpDown(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>


}

// -----

// Mul propagation test with an add at bottom which creates an upward and downward mul.
// CHECK-LABEL: @AdjustFQMulDownAdd
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulDownAdd(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.Add(%2, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>


    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[OUT0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Multiply([[INPUT_0]], [[OUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.FakeQuantize([[OUT2]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.Add([[OUT4]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT6:%.+]] = IE.Multiply([[OUT5]], %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>


    return %3 : tensor<1x128x32x64xf32>

}


// -----


// Test IE.multiply with mul propagation up and add with identical inputs propagation of Mul down.
// This tests add with identical inputs DOES NOT propagate a mul upward. Therefore FQ is updated only once.
// CHECK-LABEL: @AdjustFQMulUpAddDownSq
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulUpAddDownSq(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Add(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %2 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Add([[OUT2]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}


// -----


// Test propagation through an identical input Add with identical input multiply at the top.
// CHECK-LABEL: @AdjustFQMulDownAddSq
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulDownAddSq(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %3 : tensor<1x128x32x64xf32>
    
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Add([[OUT3]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.Multiply([[OUT4]], %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}





// -----

// Test propagation through identical input add  with identical input multiplies at top and bottom
// CHECK-LABEL: @AdjustFQMulSquareUpDownAdd
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulSquareUpDownAdd(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.FakeQuantize(%0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Multiply(%1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.Add(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %3 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Add([[OUT3]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.Multiply([[OUT4]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>


}

// -----

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

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT_0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT_1:%.+]] = IE.Multiply([[INPUT_0]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_2:%.+]] = IE.Subtract([[OUT_1]], [[OUT_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_3:%.+]] = IE.FakeQuantize([[OUT_2]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_4:%.+]] = IE.Multiply([[OUT_3]], [[OUT_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_5:%.+]] = IE.FakeQuantize([[OUT_4]], [[CST_5]], [[CST_0]], [[CST_5]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_6:%.+]] = IE.Multiply([[OUT_5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_7:%.+]] = IE.ReduceMean([[OUT_6]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT_8:%.+]] = IE.FakeQuantize([[OUT_7]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT_9:%.+]] = IE.Add([[OUT_8]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>


}

// -----

// This tests propagation of muls all the way up and bottom.
// CHECK-LABEL: @AdjustFakeQuantizeWithFQ2
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQuantizeWithFQ2(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {
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
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.Multiply(%3, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %5 = IE.FakeQuantize(%4, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %6 = IE.Add(%5, %add_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>

    return %6 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<103.136948> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>, [#const.Rescale<0.20441406965255737 : f64>]
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Subtract([[OUT1]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.FakeQuantize([[OUT2]], [[CST_4]], [[CST_3]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT2]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.FakeQuantize([[OUT4]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT6:%.+]] = IE.Multiply([[OUT5]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT7:%.+]] = IE.FakeQuantize([[OUT6]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT8:%.+]] = IE.Add([[OUT7]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT9:%.+]] = IE.Multiply([[OUT8]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}

// -----

// Test to check propagation of multiply op down with square inputs and fusion of multiply factor in
// up propagation in particular propagated mul does not reach all the way to subtract.
// Test fusion to bias.
// CHECK-LABEL: @AdjustFQMulUpFuseConstWithBias
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @AdjustFQMulUpFuseConstWithBias(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x1x32x64xf32> {
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %mul_in = const.Declare tensor<1x128x1x1xf32> = dense <8.537536814401392e-06> : tensor<1x128x1x1xf32> isSplat
    %bias_in = const.Declare tensor<1x128x1x1xf32> = dense <9.537536814401392e-06> : tensor<1x128x1x1xf32> isSplat

   
    %0 = IE.Subtract(%arg0, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.Convolution(%0, %mul_in, %bias_in) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1], auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x1x32x64xf32>
    %2 = IE.FakeQuantize(%1, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x64xf32>
    %3 = IE.Multiply(%2, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32> -> tensor<1x1x32x64xf32>
    return %3 : tensor<1x1x32x64xf32>


    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x128x1x1xf32> = dense<8.53753681E-6> : tensor<1x128x1x1xf32>, [#const.Rescale<0.20441406965255737 : f64>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x128x1x1xf32> = dense<8.53753681E-6> : tensor<1x128x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x128x1x1xf32> = dense<9.53753715E-6> : tensor<1x128x1x1xf32>, [#const.Rescale<0.20441406965255737 : f64>]
    // CHECK: [[OUT0:%.+]] = IE.Subtract([[INPUT_0]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Convolution([[OUT0]], [[CST_2]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32>, tensor<1x128x1x1xf32> -> tensor<1x1x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x1x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[OUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x64xf32>, tensor<1x1x32x64xf32> -> tensor<1x1x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x32x64xf32>, tensor<1xf32> -> tensor<1x1x32x64xf32>

}


// -----


// Testing propagation of mul to quantized weights, a case seen in PSD7 model for example.
// CHECK-LABEL: @AFQDQConvQuantizedWts
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x256x256x256xf32>
func.func @AFQDQConvQuantizedWts(%arg0 : tensor<1x256x256x256xf32>) -> tensor<1x256x256x256xf32> {
    %cst_1340 = const.Declare tensor<256x256x3x3xf32> = dense<100> : tensor<256x256x3x3xui8>, [#const.CastElemType<f32>]
    %cst_1341 = const.Declare tensor<1x1x1x1xf32> = dense<4.163311> : tensor<1x1x1x1xf32>
    %cst_1342 = const.Declare tensor<1x1x1x1xf32> = dense<-3.87944865> : tensor<1x1x1x1xf32> 
    %cst_1383 = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1384 = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

    // %16 and %564 are from before AFQDQ pass in psd7.
    %16 = IE.FakeQuantize(%cst_1340, %cst_1383, %cst_1384, %cst_1342, %cst_1341) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<256x256x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<256x256x3x3xf32> 
    %564 = IE.Convolution(%arg0, %16) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x256x256x256xf32>, tensor<256x256x3x3xf32> -> tensor<1x256x256x256xf32>
    %2000 = IE.FakeQuantize(%564, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x256x256x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x256x256x256xf32>
    return %2000 : tensor<1x256x256x256xf32>


    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<256x256x3x3xf32> = dense<100> : tensor<256x256x3x3xui8>, [#const.CastElemType<f32>]
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.79301387> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.85103935> : tensor<1x1x1x1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.FakeQuantize([[CST_1]], [[CST_2]], [[CST_3]], [[CST_4]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<256x256x3x3xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<256x256x3x3xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Convolution([[INPUT_0]], [[OUT0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x256x256x256xf32>, tensor<256x256x3x3xf32> -> tensor<1x256x256x256xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.FakeQuantize([[OUT1]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x256x256x256xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x256x256x256xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.Multiply([[OUT2]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x256x256x256xf32>, tensor<1xf32> -> tensor<1x256x256x256xf32>
}

// -----

// Tests FQ prop to multiple users.
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
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>, [#const.Rescale<4.892031192779541 : f64>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Subtract([[OUT1]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.FakeQuantize([[OUT2]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.FakeQuantize([[OUT4]], [[CST_5]], [[CST_0]], [[CST_5]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT6:%.+]] = IE.Multiply([[OUT5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT7:%.+]] = IE.ReduceMean([[OUT6]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT8:%.+]] = IE.FakeQuantize([[OUT7]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT9:%.+]] = IE.Multiply([[OUT5]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT10:%.+]] = IE.FakeQuantize([[OUT9]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>

}


// -----

// Tests FQ prop to multiple users. 
// Further tests down prop from an FQ and fusion with mul_in in the down direction.
// CHECK-LABEL: @AdjustFakeQuantizeWithMultipleUsers2
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQuantizeWithMultipleUsers2(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> (tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>) {
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
    %6 = IE.Multiply(%1, %mul_in) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %7 = IE.FakeQuantize(%6, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>

    return %5, %7 : tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>


    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<8.53753681E-6> : tensor<1x1x1x1xf32>, [#const.Rescale<4.892031192779541 : f64>]
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Subtract([[OUT1]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.FakeQuantize([[OUT2]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.FakeQuantize([[OUT4]], [[CST_5]], [[CST_0]], [[CST_5]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT6:%.+]] = IE.Multiply([[OUT5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT7:%.+]] = IE.ReduceMean([[OUT6]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT8:%.+]] = IE.FakeQuantize([[OUT7]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT9:%.+]] = IE.Multiply([[OUT3]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT10:%.+]] = IE.FakeQuantize([[OUT9]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
}

// -----

// Tests FQ prop to multiple users. 
// Further tests down prop from an FQ and  update of another FQ in the down direction.
// CHECK-LABEL: @AdjustFakeQuantizeWithMultipleUsers3
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQuantizeWithMultipleUsers3(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> (tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>) {
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
    %7 = IE.FakeQuantize(%1, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>

    return %5, %7 : tensor<1x128x1x1xf32>, tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<103.136948> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<504.549164> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_6:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT0:%.+]] = IE.Multiply([[INPUT_1]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x1x1xf32>, tensor<1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT1:%.+]] = IE.Multiply([[INPUT_0]], [[CST_6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT2:%.+]] = IE.Subtract([[OUT1]], [[OUT0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT3:%.+]] = IE.FakeQuantize([[OUT2]], [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT4:%.+]] = IE.Multiply([[OUT3]], [[OUT3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT5:%.+]] = IE.FakeQuantize([[OUT4]], [[CST_5]], [[CST_1]], [[CST_5]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT6:%.+]] = IE.Multiply([[OUT5]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT7:%.+]] = IE.ReduceMean([[OUT6]]) {axes_value = [2, 3], keep_dims} : tensor<1x128x32x64xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT8:%.+]] = IE.FakeQuantize([[OUT7]], [[CST_5]], [[CST_4]], [[CST_5]], [[CST_4]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x1x1xf32>
    // CHECK-NEXT: [[OUT9:%.+]] = IE.FakeQuantize([[OUT3]], [[CST_5]], [[CST]], [[CST_5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT10:%.+]] = IE.Multiply([[OUT9]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>

}



// -----

// CHECK-LABEL: @AdjustFakeQDQComplexAdd
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQDQComplexAdd(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.Add(%3, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %4 : tensor<1x128x32x64xf32>
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>

    // CHECK-NEXT: [[AND0:%.+]] = IE.And([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[AND0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = IE.Multiply([[MUL1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[AND3:%.+]] = IE.And([[AND0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL4:%.+]] = IE.Multiply([[AND3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL5:%.+]] = IE.Multiply([[MUL4]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[ADD6:%.+]] = IE.Add([[MUL2]], [[MUL5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ7:%.+]] = IE.FakeQuantize([[ADD6]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[ADD8:%.+]] = IE.Add([[FQ7]], [[ADD6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL9:%.+]] = IE.Multiply([[ADD8]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}

// -----

// CHECK-LABEL: @AdjustFakeQDQComplexSub
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQDQComplexSub(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.Add(%3, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    return %4 : tensor<1x128x32x64xf32>
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>

    // CHECK-NEXT: [[AND0:%.+]] = IE.And([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[AND0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = IE.Multiply([[MUL1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[AND3:%.+]] = IE.And([[AND0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL4:%.+]] = IE.Multiply([[AND3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL5:%.+]] = IE.Multiply([[MUL4]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[SUB6:%.+]] = IE.Subtract([[MUL2]], [[MUL5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ7:%.+]] = IE.FakeQuantize([[SUB6]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[ADD8:%.+]] = IE.Add([[FQ7]], [[SUB6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL9:%.+]] = IE.Multiply([[ADD8]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}


// -----

// CHECK-LABEL: @AdjustFakeQDQComplexSub2
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFakeQDQComplexSub2(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <160302.078125> : tensor<1x1x1x1xf32> isSplat

   
    %0 = IE.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.Subtract(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.Subtract(%3, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>

    return %4 : tensor<1x128x32x64xf32>
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>

    // CHECK-NEXT: [[AND0:%.+]] = IE.And([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL1:%.+]] = IE.Multiply([[AND0]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = IE.Multiply([[MUL1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[AND3:%.+]] = IE.And([[AND0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL4:%.+]] = IE.Multiply([[AND3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL5:%.+]] = IE.Multiply([[MUL4]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[SUB6:%.+]] = IE.Subtract([[MUL2]], [[MUL5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ7:%.+]] = IE.FakeQuantize([[SUB6]], [[CST_1]], [[CST_0]], [[CST_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[SUB8:%.+]] = IE.Subtract([[FQ7]], [[SUB6]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x32x64xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL9:%.+]] = IE.Multiply([[SUB8]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}


// -----


// Sandwich an overflowing FQ in between two non-overflowing FQs and make sure propagation works properly
// CHECK-LABEL: @AdjustFQSandwichNF_OF_NF
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFQSandwichNF_OF_NF(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {
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
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.FakeQuantize(%1, %fq1_in_low, %fq1_in_hi, %fq1_out_low, %fq1_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.FakeQuantize(%2, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %4 = IE.FakeQuantize(%3, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>


    return %4 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<103.136948> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK-NEXT: [[OUT_0:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_1:%.+]] = IE.And([[OUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_2:%.+]] = IE.Multiply([[OUT_1]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_3:%.+]] = IE.FakeQuantize([[OUT_2]], [[CST_4]], [[CST_3]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_4:%.+]] = IE.FakeQuantize([[OUT_3]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_5:%.+]] = IE.FakeQuantize([[OUT_4]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[OUT_6:%.+]] = IE.Multiply([[OUT_5]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}

// Sandwich + Added Reshape ops to check propagation
// CHECK-LABEL: @AdjustFQSandwichNF_OF_NF_Reshape
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>,
// CHECK-SAME:     [[INPUT_1:%.+]]: tensor<1x128x1x1xf32>
func.func @AdjustFQSandwichNF_OF_NF_Reshape(%arg0 : tensor<1x128x32x64xf32>, %arg1 : tensor<1x128x1x1xf32>) -> tensor<1x128x32x64xf32> {
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
    %1 = IE.And(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    %2 = IE.FakeQuantize(%1, %fq1_in_low, %fq1_in_hi, %fq1_out_low, %fq1_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %3 = IE.Reshape(%2) { shape_value = [1, 128, 16, 128] } : tensor<1x128x32x64xf32> -> tensor<1x128x16x128xf32>
    %4 = IE.Reshape(%3) { shape_value = [1, 128, 32, 64] } : tensor<1x128x16x128xf32> -> tensor<1x128x32x64xf32>

    %5 = IE.FakeQuantize(%4, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    %6 = IE.FakeQuantize(%5, %fq3_in_low, %fq3_in_hi, %fq3_out_low, %fq3_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>


    return %6 : tensor<1x128x32x64xf32>
    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf32> = dense<4.89203119> : tensor<1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<103.136948> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.276800e+04> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<80.2275543> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-37.2369881> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_5:%.+]] = const.Declare tensor<1xf32> = dense<0.20441407> : tensor<1xf32>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[AND1:%.+]] = IE.And([[SUB]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1x128x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL2:%.+]] = IE.Multiply([[AND1]], [[CST_5]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ3:%.+]] = IE.FakeQuantize([[MUL2]], [[CST_4]], [[CST_3]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[RS4:%.+]] = IE.Reshape([[FQ3]]) {shape_value = [1, 128, 16, 128]} : tensor<1x128x32x64xf32> -> tensor<1x128x16x128xf32>
    // CHECK-NEXT: [[RS5:%.+]] = IE.Reshape([[RS4]]) {shape_value = [1, 128, 32, 64]} : tensor<1x128x16x128xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ6:%.+]] = IE.FakeQuantize([[RS5]], [[CST_2]], [[CST_1]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[FQ7:%.+]] = IE.FakeQuantize([[FQ6]], [[CST_2]], [[CST_0]], [[CST_2]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NEXT: [[MUL8:%.+]] = IE.Multiply([[FQ7]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x64xf32>, tensor<1xf32> -> tensor<1x128x32x64xf32>
}

// -----

// Asymmetric FQ is skipped and handled by HandleU16FakeQuantize using ScaleShift Op
// CHECK-LABEL: @SkipAsymmetricFakeQuantizeOp
// CHECK-SAME:     [[INPUT_0:%.+]]: tensor<1x128x32x64xf32>
func.func @SkipAsymmetricFakeQuantizeOp(%arg0 : tensor<1x128x32x64xf32>) -> tensor<1x128x32x64xf32> {

    %fq2_in_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_in_hi = const.Declare tensor<1x1x1x1xf32> = dense <1.0> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_low = const.Declare tensor<1x1x1x1xf32> = dense <0.000000e+00> : tensor<1x1x1x1xf32> isSplat
    %fq2_out_hi = const.Declare tensor<1x1x1x1xf32> = dense <6.5535e+04> : tensor<1x1x1x1xf32> isSplat

    %1 = IE.FakeQuantize(%arg0, %fq2_in_low, %fq2_in_hi, %fq2_out_low, %fq2_out_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    return %1 : tensor<1x128x32x64xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.553500e+04> : tensor<1x1x1x1xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-DAG: [[FQ:%.+]] = IE.FakeQuantize([[INPUT_0]], [[CST]], [[CST_0]], [[CST]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1x128x32x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x128x32x64xf32>
    // CHECK-NOT: IE.Multiply
}

// Test to check that FakeQuantize with NaN parameters is handled gracefully
// CHECK-LABEL: @AdjustFakeQuantizeWithNaNParams
func.func @AdjustFakeQuantizeWithNaNParams() -> tensor<1xf32> {
    %cst_1 = const.Declare tensor<1xf32> = dense<1.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %cst_2 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %cst_nan = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>, [#const.Rescale<0x7FF8000000000000 : f64>]
    
    %1 = IE.FakeQuantize(%cst_1, %cst_nan, %cst_nan, %cst_nan, %cst_nan) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Multiply(%1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>

    return %2 : tensor<1xf32>

    // CHECK-DAG: [[CST_INPUT:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<f32>, [#const.Reshape<[1]>]
    // CHECK-DAG: [[CST_MUL:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK-DAG: [[CST_NAN:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>, [#const.Rescale<0x7FF8000000000000 : f64>]

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[CST_INPUT]], [[CST_NAN]], [[CST_NAN]], [[CST_NAN]], [[CST_NAN]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    // CHECK-NEXT: [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<1xf32>
}

// -----

// CHECK-LABEL: @AdjustFakeQuantizeWithNaNParamsNonSplat
func.func @AdjustFakeQuantizeWithNaNParamsNonSplat() -> tensor<2xf32> {
    %cst_1 = const.Declare tensor<2xf32> = dense<[1.0, 2.0]> : tensor<2xf32>
    %cst_2 = const.Declare tensor<2xf32> = dense<[3.0, 4.0]> : tensor<2xf32>
    %cst_non_splat = const.Declare tensor<2xf32> = dense<[2.0, 3.0]> : tensor<2xf32>, [#const.Rescale<Content<dense<[0x7FF8000000000000, 4.0]> : tensor<2xf64>>>]

    %1 = IE.FakeQuantize(%cst_1, %cst_non_splat, %cst_non_splat, %cst_non_splat, %cst_non_splat) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
    %2 = IE.Multiply(%1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>

    return %2 : tensor<2xf32>

    // CHECK-DAG: [[CST_INPUT:%.+]] = const.Declare tensor<2xf32> = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    // CHECK-DAG: [[CST_MUL:%.+]] = const.Declare tensor<2xf32> = dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
    // CHECK-DAG: [[CST_NON_SPLAT:%.+]] = const.Declare tensor<2xf32> = dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>, [#const.Rescale<Content<dense<[0x7FF8000000000000, 4.000000e+00]> : tensor<2xf64>>>]

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[CST_INPUT]], [[CST_NON_SPLAT]], [[CST_NON_SPLAT]], [[CST_NON_SPLAT]], [[CST_NON_SPLAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 65536 : i64} : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
    // CHECK-NEXT: [[MUL:%.+]] = IE.Multiply([[FQ]], [[CST_MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
    // CHECK-NEXT: return [[MUL]] : tensor<2xf32>
}
