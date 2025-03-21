//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-rmsnorm --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @FuseRMSNorm
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNorm(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x3072xf32> = dense<1.000000e+00> : tensor<1x1x3072xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %1 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %3 = IE.Power(%2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %4 = IE.ReduceMean(%3) {axes_value = [2], keep_dims} : tensor<1x1x3072xf32> -> tensor<1x1x1xf32>
    %5 = IE.Add(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %6 = IE.Sqrt(%5) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %7 = IE.Divide(%cst_1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %8 = IE.Multiply(%2, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %9 = IE.Multiply(%8, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %10 = IE.Convert(%9) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    return %10 : tensor<1x1x3072xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<1x1x3072xf32>, [#const.Reshape<[3072]>]
    // CHECK: [[CONVERT0:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[ARG1]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONVERT0]], [[CONVERT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[RMS:%.+]] = IE.RMS([[ADD]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[RMS]]) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    // CHECK: return [[CONVERT2]] : tensor<1x1x3072xf16>
}

// -----

// CHECK-LABEL: @FuseRMSNormConvert
func.func @FuseRMSNormConvert(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x3072xf16> = dense<1.000000e+00> : tensor<1x1x3072xf16>
    %add = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf16>, tensor<1x1x3072xf16> -> tensor<1x1x3072xf16>
    %convert = IE.Convert(%add) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %power = IE.Power(%convert, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %rm = IE.ReduceMean(%power) {axes_value = [2], keep_dims} : tensor<1x1x3072xf32> -> tensor<1x1x1xf32>
    %add2 = IE.Add(%rm, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %sqrt = IE.Sqrt(%add2) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %div = IE.Divide(%cst_1, %sqrt) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %mult1 = IE.Multiply(%convert, %div) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %convert2 = IE.Convert(%mult1) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    %mult2 = IE.Multiply(%convert2, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf16>, tensor<1x1x3072xf16> -> tensor<1x1x3072xf16>
    return %mult2 : tensor<1x1x3072xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3072xf16> = dense<1.000000e+00> : tensor<1x1x3072xf16>, [#const.Reshape<[3072]>]
    // CHECK: [[ADD:%.+]] = IE.Add
    // CHECK-NOT: IE.Convert
    // CHECK: [[RMS:%.+]] = IE.RMS([[ADD]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x3072xf16>, tensor<3072xf16> -> tensor<1x1x3072xf16>
    // CHECK: return [[RMS]] : tensor<1x1x3072xf16>
}

// -----

// CHECK-LABEL: @FuseRMSNormConstInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1xf16>
func.func @FuseRMSNormConstInput(%arg0: tensor<1x1xf16>) -> tensor<1x1x3584xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<9.99999997E-7> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x3584xf32> = dense<1.0> : tensor<1x1x3584xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 1, 3584]} : tensor<1x1xf32> -> tensor<1x1x3584xf32>
    %2 = IE.Power(%1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3584xf32>, tensor<1x1x1xf32> -> tensor<1x1x3584xf32>
    %3 = IE.ReduceMean(%2) {axes_value = [2], keep_dims} : tensor<1x1x3584xf32> -> tensor<1x1x1xf32>
    %4 = IE.Add(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %5 = IE.Sqrt(%4) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %6 = IE.Divide(%cst_1, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %7 = IE.Multiply(%1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3584xf32>, tensor<1x1x1xf32> -> tensor<1x1x3584xf32>
    %8 = IE.Multiply(%7, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3584xf32>, tensor<1x1x3584xf32> -> tensor<1x1x3584xf32>
    %9 = IE.Convert(%8) {dstElemType = f16} : tensor<1x1x3584xf32> -> tensor<1x1x3584xf16>
    return %9 : tensor<1x1x3584xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3584xf32> = dense<1.000000e+00> : tensor<1x1x3584xf32>, [#const.Reshape<[3584]>]
    // CHECK: [[CONVERT0:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[CONVERT0]]) {shape_value = [1, 1, 3584]} : tensor<1x1xf32> -> tensor<1x1x3584xf32>
    // CHECK: [[RMS:%.+]] = IE.RMS([[RESHAPE]], [[CST]]) {epsilon = 9.9999999747524271E-7 : f64} : tensor<1x1x3584xf32>, tensor<3584xf32> -> tensor<1x1x3584xf32>
    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[RMS]]) {dstElemType = f16} : tensor<1x1x3584xf32> -> tensor<1x1x3584xf16>
    // CHECK: return [[CONVERT1]] : tensor<1x1x3584xf16>
}

// -----

// CHECK-LABEL: @FuseRMSNormPrefill
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x3072xf32>, [[ARG1:%.+]]: tensor<1x1024x3072xf32>)
func.func @FuseRMSNormPrefill(%arg0: tensor<1x1024x3072xf32>, %arg1: tensor<1x1024x3072xf32>) -> tensor<1x1024x3072xf32> {
  %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1x3072xf32> = dense<1.0> : tensor<1x1x3072xf32>
  %0 = IE.Add(%arg1, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x3072xf32> -> tensor<1x1024x3072xf32>
  %1 = IE.Power(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1024x3072xf32>
  %2 = IE.ReduceMean(%1) {axes_value = [2], keep_dims} : tensor<1x1024x3072xf32> -> tensor<1x1024x1xf32>
  %3 = IE.Add(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1xf32>
  %4 = IE.Sqrt(%3) : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
  %5 = IE.Divide(%cst_1, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
  %6 = IE.Multiply(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x3072xf32>
  %7 = IE.Multiply(%6, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1024x3072xf32>
  return %7 : tensor<1x1024x3072xf32>

  // CHECK:  [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<1x1x3072xf32>, [#const.Reshape<[3072]>]
  // CHECK:  [[ADD:%.+]] = IE.Add([[ARG1]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x3072xf32> -> tensor<1x1024x3072xf32>
  // CHECK:  [[RMS:%.+]] = IE.RMS([[ADD]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1024x3072xf32>, tensor<3072xf32> -> tensor<1x1024x3072xf32>
  // CHECK:  return [[RMS]] : tensor<1x1024x3072xf32>
}


// -----

// CHECK-LABEL: @FuseRMSNormCreateGama
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNormCreateGama(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %1 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %3 = IE.Power(%2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %4 = IE.ReduceMean(%3) {axes_value = [2], keep_dims} : tensor<1x1x3072xf32> -> tensor<1x1x1xf32>
    %5 = IE.Add(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %6 = IE.Sqrt(%5) : tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %7 = IE.Divide(%cst_1, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %8 = IE.Multiply(%2, %7) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %9 = IE.Convert(%8) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    return %9 : tensor<1x1x3072xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<1x1x3072xf32>, [#const.Reshape<[3072]>]
    // CHECK: [[CONVERT0:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK: [[CONVERT1:%.+]] = IE.Convert([[ARG1]]) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[CONVERT0]], [[CONVERT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[RMS:%.+]] = IE.RMS([[ADD]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[RMS]]) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    // CHECK: return [[CONVERT2]] : tensor<1x1x3072xf16>
}

// -----

// CHECK-LABEL: @IllegalFuseRMSNorm
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x3072xf32>, [[ARG1:%.+]]: tensor<1x1024x3072xf32>)
func.func @IllegalFuseRMSNorm(%arg0: tensor<1x1024x3072xf32>, %arg1: tensor<1x1024x3072xf32>) -> tensor<1x1024x3072xf32> {
  %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
  %cst_2 = const.Declare tensor<1x1024x1xf32> = dense<2.0> : tensor<1x1024x1xf32>
  %0 = IE.Add(%arg1, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x3072xf32> -> tensor<1x1024x3072xf32>
  %1 = IE.Power(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1024x3072xf32>
  %2 = IE.ReduceMean(%1) {axes_value = [2], keep_dims} : tensor<1x1024x3072xf32> -> tensor<1x1024x1xf32>
  %3 = IE.Add(%2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1xf32>
  %4 = IE.Sqrt(%3) : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
  %5 = IE.Divide(%cst_1, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
  %6 = IE.Multiply(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x3072xf32>
  %7 = IE.Multiply(%6, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x3072xf32>
  return %7 : tensor<1x1024x3072xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST2:%.+]] = const.Declare tensor<1x1024x1xf32> = dense<2.000000e+00> : tensor<1x1024x1xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[ARG1]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x3072xf32> -> tensor<1x1024x3072xf32>
    // CHECK: [[POW:%.+]] = IE.Power([[ADD]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1024x3072xf32>
    // CHECK: [[REDUCE:%.+]] = IE.ReduceMean([[POW]]) {axes_value = [2], keep_dims} : tensor<1x1024x3072xf32> -> tensor<1x1024x1xf32>
    // CHECK: [[ADD2:%.+]] = IE.Add([[REDUCE]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1xf32>
    // CHECK: [[SQRT:%.+]] = IE.Sqrt([[ADD2]]) : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CST1]], [[SQRT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[ADD]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x3072xf32>
    // CHECK: [[MULGAMMA:%.+]] = IE.Multiply([[MUL]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x3072xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x3072xf32>
    // CHECK: return [[MULGAMMA]] : tensor<1x1024x3072xf32>

}

// -----

// CHECK-LABEL: @FuseRMSNormPSU
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormPSU(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<55.4256248> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %4 = IE.Multiply(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  return %4 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormPSUUnstripped
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormPSUUnstripped(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<55.4256248> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  %0 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %4 = IE.FakeQuantize(%3, %cst_0, %cst_0, %cst_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  %5 = IE.Multiply(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  %6 = IE.FakeQuantize(%5, %cst_0, %cst_0, %cst_0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  return %6 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[RMS]], [[CST_0]], [[CST_0]], [[CST_0]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: return [[FQ]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormExtendedPattern
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x2048xf32>
func.func @FuseRMSNormExtendedPattern(%arg0: tensor<1x1x2048xf32>) -> tensor<1x1x2048xf32> {
    %cst_0 = const.Declare tensor<1x1x2048xf32> = dense<2.000000e+00> : tensor<1x1x2048xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<-0.5> : tensor<1x1x1xf32> isSplat
    %cst_2 = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32> isSplat

    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [2], keep_dims} : tensor<1x1x2048xf32> -> tensor<1x1x1xf32>
    %2 = IE.Add(%1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %3 = IE.Power(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %4 = IE.Multiply(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x1xf32> -> tensor<1x1x2048xf32>
    %5 = IE.Multiply(%4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>
    return %5 : tensor<1x1x2048xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<2048xf32> = dense<2.000000e+00> : tensor<1x1x2048xf32>, [#const.Reshape<[2048]>]
    // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x2048xf32>, tensor<2048xf32> -> tensor<1x1x2048xf32>
    // CHECK: return [[RMS]] : tensor<1x1x2048xf32>
}

// -----

// CHECK-LABEL: @NotFuseRMSNormExtendedPatternDueToUnmatchedReciprocalOfSqrt
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x2048xf32>
func.func @NotFuseRMSNormExtendedPatternDueToUnmatchedReciprocalOfSqrt(%arg0: tensor<1x1x2048xf32>) -> tensor<1x1x2048xf32> {
    %cst_0 = const.Declare tensor<1x1x2048xf32> = dense<2.000000e+00> : tensor<1x1x2048xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<2.0> : tensor<1x1x1xf32> isSplat
    %cst_2 = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32> isSplat

    %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>
    %1 = IE.ReduceMean(%0) {axes_value = [2], keep_dims} : tensor<1x1x2048xf32> -> tensor<1x1x1xf32>
    %2 = IE.Add(%1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %3 = IE.Power(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %4 = IE.Multiply(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x1xf32> -> tensor<1x1x2048xf32>
    %5 = IE.Multiply(%4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>
    return %5 : tensor<1x1x2048xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x2048xf32> = dense<2.000000e+00> : tensor<1x1x2048xf32>
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    // CHECK: [[MULTIPLY_0:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>
    // CHECK: [[REDUCE:%.+]] = IE.ReduceMean([[MULTIPLY_0]]) {axes_value = [2], keep_dims} : tensor<1x1x2048xf32> -> tensor<1x1x1xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[REDUCE]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    // CHECK: [[POWER:%.+]] = IE.Power([[ADD]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[ARG0]], [[POWER]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x1xf32> -> tensor<1x1x2048xf32>
    // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2048xf32>, tensor<1x1x2048xf32> -> tensor<1x1x2048xf32>

    // CHECK: return [[MULTIPLY_2]] : tensor<1x1x2048xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormExtendedPatternWithReduceSum
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormExtendedPatternWithReduceSum(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<55.4256248> : tensor<1x1x1x1xf32>
  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %4 = IE.Multiply(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  return %4 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}
