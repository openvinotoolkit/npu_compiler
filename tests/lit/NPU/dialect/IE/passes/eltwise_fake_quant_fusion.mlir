//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --eltwise-fake-quantize-fusion %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @AddFakeQuantizeFusionScalarConstLhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @AddFakeQuantizeFusionScalarConstLhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<25.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Add(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[GELU]], [[CST_0]], [[CST]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @SubtractFakeQuantizeFusionScalarConstLhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @SubtractFakeQuantizeFusionScalarConstLhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<5.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-10.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Subtract(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e+00> : tensor<1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[SUB:%.+]] = IE.Subtract([[CST]], [[GELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[SUB]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @SubtractFakeQuantizeFusionScalarConstRhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @SubtractFakeQuantizeFusionScalarConstRhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<5.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-10.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Subtract(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x32x32xf32>, tensor<1xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.700000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK-NOT:   IE.Subtract
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[GELU]], [[CST_0]], [[CST]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @MultiplyFakeQuantizeFusionScalarConstLhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @MultiplyFakeQuantizeFusionScalarConstLhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<3.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Multiply(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.666666686> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<4.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK-NOT:   IE.Multiply
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[GELU]], [[CST]], [[CST_0]], [[CST_1]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// E#129083
// CHECK-LABEL: @DoNotMultiplyFakeQuantizeFusionScalarConstLhsWithFQConst
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @DoNotMultiplyFakeQuantizeFusionScalarConstLhsWithFQConst(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.00> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.FakeQuantize(%cst, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    %1 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %2 = IE.Multiply(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %3 = IE.FakeQuantize(%2, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %3 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[FQ1:%.+]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[FQ1]], [[GELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[FQ2:%.+]] = IE.FakeQuantize([[MUL]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ2]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @DivideFakeQuantizeFusionScalarConstLhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @DivideFakeQuantizeFusionScalarConstLhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<3.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Divide(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf32> = dense<3.000000e+00> : tensor<1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[DIV:%.+]] = IE.Divide([[CST]], [[GELU]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[DIV]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @DivideFakeQuantizeFusionScalarConstRhs
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @DivideFakeQuantizeFusionScalarConstRhs(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<3.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Gelu(%arg0) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Divide(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x32x32xf32>, tensor<1xf32> -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<3.600000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       [[GELU:%.+]] = IE.Gelu([[INPUT]]) : tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK-NOT:   IE.Divide
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[GELU]], [[CST_0]], [[CST]], [[CST_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @DoNotAddFakeQuantizeFusionDPUParentOp
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @DoNotAddFakeQuantizeFusionDPUParentOp(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<25.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %filter = const.Declare tensor<12x12x1x1xf32> = dense<1.0> : tensor<12x12x1x1xf32>
    %0 = IE.Convolution(%arg0, %filter)
      {
          strides = [1, 1],
          pads_begin = [0, 0],
          pads_end = [0, 0],
          dilations = [1, 1]
      } :
      tensor<1x12x32x32xf32>, tensor<12x12x1x1xf32> -> tensor<1x12x32x32xf32>
    %1 = IE.Add(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1xf32> = dense<2.500000e+01> : tensor<1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[FILTER:%.+]] = const.Declare tensor<12x12x1x1xf32> = dense<1.000000e+00> : tensor<12x12x1x1xf32>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[INPUT]], [[FILTER]]) 
    // CHECK-SAME       tensor<1x12x32x32xf32>, tensor<12x12x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[CONV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x32x32xf32>, tensor<1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ADD_0]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}

// -----

// CHECK-LABEL: @DoNotAddFakeQuantizeFusionNonScalarConst
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x4x32x32xf32>
func.func @DoNotAddFakeQuantizeFusionNonScalarConst(%arg0: tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32> {
    %cst = const.Declare tensor<1x4x1x1xf32> = dense<[[[[0.00]], [[2.00]], [[5.2]], [[6.5]]]]> : tensor<1x4x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x4x32x32xf32> -> tensor<1x4x32x32xf32>
    %1 = IE.Add(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1x1xf32>, tensor<1x4x32x32xf32>  -> tensor<1x4x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x32x32xf32>
    return %2 : tensor<1x4x32x32xf32>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x4x1x1xf32>
    // CHECK-SAME{LITERAL}: = dense<[[[[0.000000e+00]], [[2.000000e+00]], [[5.200000e+00]], [[6.500000e+00]]]]> : tensor<1x4x1x1xf32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[INPUT]]) {max = 3.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x4x32x32xf32> -> tensor<1x4x32x32xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CLAMP]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x32x32xf32>, tensor<1x4x1x1xf32>  -> tensor<1x4x32x32xf32>
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ADD]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x4x32x32xf32>
}

// -----

// CHECK-LABEL: @AddFakeQuantizeFusionDPUParentOp
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @AddFakeQuantizeFusionDPUParentOp(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<25.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %0 = IE.Add(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x32x32xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %1 = IE.Add(%cst, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK:       [[ADD_0:%.+]] = IE.Add([[INPUT]], [[INPUT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x32x32xf32>, tensor<1x12x32x32xf32> -> tensor<1x12x32x32xf32>
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[ADD_0]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}


// -----

// CHECK-LABEL: @AddFakeQuantizeFusion
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x12x32x32xf32>
func.func @AddFakeQuantizeFusion(%arg0: tensor<1x12x32x32xf32>) -> tensor<1x12x32x32xf32> {
    %cst = const.Declare tensor<1xf32> = dense<25.00> : tensor<1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<12.00> : tensor<1x1x1x1xf32>
    %1 = IE.Add(%cst, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x12x32x32xf32>  -> tensor<1x12x32x32xf32>
    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    return %2 : tensor<1x12x32x32xf32>

    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-2.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.300000e+01> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.200000e+01> : tensor<1x1x1x1xf32>
    // CHECK-NOT:   IE.Add
    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x12x32x32xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x12x32x32xf32>
    // CHECK:       return [[FQ]] : tensor<1x12x32x32xf32>
}
