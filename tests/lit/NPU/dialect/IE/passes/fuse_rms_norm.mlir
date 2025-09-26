//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-rmsnorm --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @FuseRMSNormForReduceMean
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNormForReduceMean(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanMaximumSqrt
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNormForReduceMeanMaximumSqrt(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1xf32>
    %cst_2 = const.Declare tensor<1x1x3072xf32> = dense<1.000000e+00> : tensor<1x1x3072xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %1 = IE.Convert(%arg1) {dstElemType = f32} : tensor<1x1x3072xf16> -> tensor<1x1x3072xf32>
    %2 = IE.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %3 = IE.Power(%2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x1xf32> -> tensor<1x1x3072xf32>
    %4 = IE.ReduceMean(%3) {axes_value = [2], keep_dims} : tensor<1x1x3072xf32> -> tensor<1x1x1xf32>
    %5 = IE.Maximum(%4, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x1x1xf32>
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanFixSmallEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNormForReduceMeanFixSmallEpsilon(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
    %cst = const.Declare tensor<1x1x1xf32> = dense<1.0000000E-12> : tensor<1x1x1xf32>
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
    // CHECK: [[RMS:%.+]] = IE.RMS([[ADD]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: [[CONVERT2:%.+]] = IE.Convert([[RMS]]) {dstElemType = f16} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf16>
    // CHECK: return [[CONVERT2]] : tensor<1x1x3072xf16>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceMeanConvert
func.func @FuseRMSNormForReduceMeanConvert(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanConstInput
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1xf16>
func.func @FuseRMSNormForReduceMeanConstInput(%arg0: tensor<1x1xf16>) -> tensor<1x1x3584xf16> {
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanPrefill
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x3072xf32>, [[ARG1:%.+]]: tensor<1x1024x3072xf32>)
func.func @FuseRMSNormForReduceMeanPrefill(%arg0: tensor<1x1024x3072xf32>, %arg1: tensor<1x1024x3072xf32>) -> tensor<1x1024x3072xf32> {
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanCreateGama
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf16>, [[ARG1:%.+]]: tensor<1x1x3072xf16>)
func.func @FuseRMSNormForReduceMeanCreateGama(%arg0: tensor<1x1x3072xf16>, %arg1: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
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

// CHECK-LABEL: @IllegalFuseRMSNormForReduceMean
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x3072xf32>, [[ARG1:%.+]]: tensor<1x1024x3072xf32>)
func.func @IllegalFuseRMSNormForReduceMean(%arg0: tensor<1x1024x3072xf32>, %arg1: tensor<1x1024x3072xf32>) -> tensor<1x1024x3072xf32> {
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

// CHECK-LABEL: @FuseRMSNormForReduceSumPSU
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormForReduceSumPSU(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
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

// CHECK-LABEL: @FuseRMSNormForReduceSumPSUUnstripped
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormForReduceSumPSUUnstripped(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
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

// CHECK-LABEL: @FuseRMSNormForReduceMeanExtendedPattern
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x2048xf32>
func.func @FuseRMSNormForReduceMeanExtendedPattern(%arg0: tensor<1x1x2048xf32>) -> tensor<1x1x2048xf32> {
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

// CHECK-LABEL: @NotFuseRMSNormForReduceMeanExtendedPatternDueToUnmatchedReciprocalOfSqrt
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x1x2048xf32>
func.func @NotFuseRMSNormForReduceMeanExtendedPatternDueToUnmatchedReciprocalOfSqrt(%arg0: tensor<1x1x2048xf32>) -> tensor<1x1x2048xf32> {
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

// -----

// CHECK-LABEL: @FuseRMSNormWithReduceSumArbitraryScaleMult
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormWithReduceSumArbitraryScaleMult(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<8.123456> : tensor<1x1x1x1xf32>
  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %4 = IE.Multiply(%3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>
  return %4 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<0.14656499> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormWithReduceSumSkipFoldingNonSplatScaleMult
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x30x10xf32>)
func.func @FuseRMSNormWithReduceSumSkipFoldingNonSplatScaleMult(%arg0: tensor<1x1x30x10xf32>) -> tensor<1x1x30x10xf32> {
  %scale = const.Declare tensor<1x1x1x10xf32> = dense<[[[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]]]> : tensor<1x1x1x10xf32>
  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x10xf32>, tensor<1x1x30x10xf32> -> tensor<1x1x30x10xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x30x10xf32> -> tensor<1x1x30x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x30x1xf32> -> tensor<1x1x30x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x10xf32>, tensor<1x1x30x1xf32> -> tensor<1x1x30x10xf32>
  %4 = IE.Multiply(%3, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x10xf32>, tensor<1x1x1x10xf32> -> tensor<1x1x30x10xf32>
  return %4 : tensor<1x1x30x10xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<10xf32> = dense<0.316227764> : tensor<10xf32>
  // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x10xf32> = dense<{{\[\[\[\[}}1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 0.699999988, 8.000000e-01, 0.899999976, 1.000000e+00{{\]\]\]\]}}> : tensor<1x1x1x10xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x30x10xf32>, tensor<10xf32> -> tensor<1x1x30x10xf32>
  // CHECK: [[MULT:%.+]] = IE.Multiply([[RMS]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x30x10xf32>, tensor<1x1x1x10xf32> -> tensor<1x1x30x10xf32>
  // CHECK: return [[MULT]] : tensor<1x1x30x10xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumPatternUpdateEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormForReduceSumPatternUpdateEpsilon(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<9.99999993E-9> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<55.4256248> : tensor<1x1x1x1xf32>

  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Sqrt(%2) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %4 = IE.Divide(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %5 = IE.Multiply(%4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>

  return %5 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999999392252903E-9 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>

  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumPatternWithDivideOne
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormForReduceSumPatternWithDivideOne(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<55.4256248> : tensor<1x1x1x1xf32>

  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%cst, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %4 = IE.Multiply(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>
  %5 = IE.Multiply(%4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x3072xf32>

  return %5 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<1.000000e+00> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>

  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumPatternWithoutScale
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @FuseRMSNormForReduceSumPatternWithoutScale(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>

  return %3 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<3072xf32> = dense<0.0180421956> : tensor<3072xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x512x3072xf32>, tensor<3072xf32> -> tensor<1x1x512x3072xf32>

  // CHECK: return [[RMS]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumComplexPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x512x512x3x3xf32>)
func.func @FuseRMSNormForReduceSumComplexPattern(%arg0: tensor<1x512x512x3x3xf32>) -> tensor<1x512x512x3x3xf32> {
  %cst = const.Declare tensor<1x1xf32> = dense<9.99999993E-9> : tensor<1x1xf32>
  %cst_0 = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>

  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x512x3x3xf32>, tensor<1x512x512x3x3xf32> -> tensor<1x512x512x3x3xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [2, 3, 4]} : tensor<1x512x512x3x3xf32> -> tensor<1x512xf32>
  %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512xf32>, tensor<1x1xf32> -> tensor<1x512xf32>
  %3 = IE.Sqrt(%2) : tensor<1x512xf32> -> tensor<1x512xf32>
  %4 = IE.Divide(%cst_0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xf32>, tensor<1x512xf32> -> tensor<1x512xf32>
  %5 = IE.AffineReshape(%4) {dim_mapping = [[0], [1, 2, 3, 4]], shape_value = [1, 512, 1, 1, 1]} : tensor<1x512xf32> -> tensor<1x512x1x1x1xf32>
  %6 = IE.Multiply(%arg0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x512x512x3x3xf32>, tensor<1x512x1x1x1xf32> -> tensor<1x512x512x3x3xf32>

  return %6 : tensor<1x512x512x3x3xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<4608xf32> = dense<0.0147313923> : tensor<4608xf32>
  // CHECK: [[AFFINERESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK: [[RMS:%.+]] = IE.RMS([[AFFINERESHAPE_IN]], [[CST]]) {epsilon = 9.9999999392252903E-9 : f64} : tensor<1x1x512x4608xf32>, tensor<4608xf32> -> tensor<1x1x512x4608xf32>
  // CHECK: [[AFFINERESHAPE_OUT:%.+]] = IE.AffineReshape([[RMS]])

  // CHECK: return [[AFFINERESHAPE_OUT]] : tensor<1x512x512x3x3xf32>
}

// -----

// CHECK-LABEL: @NotFuseRMSNormForReduceSumPatternAsNotMostOuterReduceAxes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x256x512xf32>)
func.func @NotFuseRMSNormForReduceSumPatternAsNotMostOuterReduceAxes(%arg0: tensor<1x40x256x512xf32>) -> tensor<1x40x256x512xf32> {
  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x40x256x512xf32> -> tensor<1x40x256x512xf32>
  %1 = IE.ReduceSum(%0) {axes_value = [1], keep_dims} : tensor<1x40x256x512xf32> -> tensor<1x1x256x512xf32>
  %2 = IE.Sqrt(%1) : tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  %3 = IE.Divide(%arg0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x1x256x512xf32> -> tensor<1x40x256x512xf32>

  return %3 : tensor<1x40x256x512xf32>

  // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x40x256x512xf32> -> tensor<1x40x256x512xf32>
  // CHECK: [[REDUCESUM:%.+]] = IE.ReduceSum([[MULTIPLY]]) {axes_value = [1], keep_dims} : tensor<1x40x256x512xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[SQRT:%.+]] = IE.Sqrt([[REDUCESUM]]) : tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[DIVIDE:%.+]] = IE.Divide([[ARG0]], [[SQRT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x1x256x512xf32> -> tensor<1x40x256x512xf32>

  // CHECK: return [[DIVIDE]] : tensor<1x40x256x512xf32>
}

// -----

// CHECK-LABEL: @NotFuseRMSNormForReduceMeanPatternAsNotMostOuterReduceAxes
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x40x256x512xf32>)
func.func @NotFuseRMSNormForReduceMeanPatternAsNotMostOuterReduceAxes(%arg0: tensor<1x40x256x512xf32>) -> tensor<1x40x256x512xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>

  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x40x256x512xf32> -> tensor<1x40x256x512xf32>
  %1 = IE.ReduceMean(%0) {axes_value = [1], keep_dims} : tensor<1x40x256x512xf32> -> tensor<1x1x256x512xf32>
  %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x256x512xf32>
  %3 = IE.Sqrt(%2) : tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  %4 = IE.Divide(%cst_0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  %5 = IE.Multiply(%arg0, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x1x256x512xf32> -> tensor<1x40x256x512xf32>

  return %5 : tensor<1x40x256x512xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1x1xf32>
  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

  // CHECK: [[MULTIPLY_IN:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x40x256x512xf32> -> tensor<1x40x256x512xf32>
  // CHECK: [[REDUCEMEAN:%.+]] = IE.ReduceMean([[MULTIPLY_IN]]) {axes_value = [1], keep_dims} : tensor<1x40x256x512xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[ADD:%.+]] = IE.Add([[REDUCEMEAN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x512xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[SQRT:%.+]] = IE.Sqrt([[ADD]]) : tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CST_0]], [[SQRT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x256x512xf32> -> tensor<1x1x256x512xf32>
  // CHECK: [[MULTIPLY_OUT:%.+]] = IE.Multiply([[ARG0]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x40x256x512xf32>, tensor<1x1x256x512xf32> -> tensor<1x40x256x512xf32>

  // CHECK: return [[MULTIPLY_OUT]] : tensor<1x40x256x512xf32>
}

// -----

// CHECK-LABEL: @NotFuseRMSNormForReduceMeanPatternAsDivideNumeratorNotOne
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x512x3072xf32>)
func.func @NotFuseRMSNormForReduceMeanPatternAsDivideNumeratorNotOne(%arg0: tensor<1x1x512x3072xf32>) -> tensor<1x1x512x3072xf32> {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1x1xf32>
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>

  %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  %2 = IE.Add(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x1xf32>
  %3 = IE.Sqrt(%2) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %4 = IE.Divide(%cst_0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  %5 = IE.Multiply(%arg0, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>

  return %5 : tensor<1x1x512x3072xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.00135803E-5> : tensor<1x1x1x1xf32>
  // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

  // CHECK: [[MULTIPLY_IN:%.+]] = IE.Multiply([[ARG0]], [[ARG0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x3072xf32> -> tensor<1x1x512x3072xf32>
  // CHECK: [[REDUCEMEAN:%.+]] = IE.ReduceMean([[MULTIPLY_IN]]) {axes_value = [3], keep_dims} : tensor<1x1x512x3072xf32> -> tensor<1x1x512x1xf32>
  // CHECK: [[ADD:%.+]] = IE.Add([[REDUCEMEAN]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x512x1xf32>
  // CHECK: [[SQRT:%.+]] = IE.Sqrt([[ADD]]) : tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  // CHECK: [[DIVIDE:%.+]] = IE.Divide([[CST_0]], [[SQRT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x1xf32>
  // CHECK: [[MULTIPLY_OUT:%.+]] = IE.Multiply([[ARG0]], [[DIVIDE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x512x3072xf32>, tensor<1x1x512x1xf32> -> tensor<1x1x512x3072xf32>

  // CHECK: return [[MULTIPLY_OUT]] : tensor<1x1x512x3072xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceMeanEndsWithDivide
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x1792xf32>)
func.func @FuseRMSNormForReduceMeanEndsWithDivide(%arg0: tensor<1x1024x1792xf32>) -> tensor<1x1024x1792xf32> {
  %cst = const.Declare tensor<1x1x1xf32> = dense<2.0> : tensor<1x1x1xf32> isSplat
  %cst_1 = const.Declare tensor<1x1x1xf32> = dense<1.0> : tensor<1x1x1xf32> isSplat

  %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1792xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1792xf32>
  %1 = IE.ReduceMean(%0) {axes_value = [2], keep_dims} : tensor<1x1024x1792xf32> -> tensor<1x1024x1xf32>
  %2 = IE.Add(%1, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1xf32>, tensor<1x1x1xf32> -> tensor<1x1024x1xf32>
  %3 = IE.Sqrt(%2) : tensor<1x1024x1xf32> -> tensor<1x1024x1xf32>
  %4 = IE.Divide(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1792xf32>, tensor<1x1024x1xf32> -> tensor<1x1024x1792xf32>

  return %4 : tensor<1x1024x1792xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1792xf32> = dense<1.000000e+00> : tensor<1792xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 1.000000e+00 : f64} : tensor<1x1024x1792xf32>, tensor<1792xf32> -> tensor<1x1024x1792xf32>

  // CHECK: return [[RMS]] : tensor<1x1024x1792xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceMeanEndsWithDivideMultiUser
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1024x1x256xf32>)
func.func @FuseRMSNormForReduceMeanEndsWithDivideMultiUser(%arg0: tensor<1x1024x1x256xf32>) -> (tensor<1x1024x1x128xf32>, tensor<1x1024x1x128xf32>) {
  %cst = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32> isSplat
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32> isSplat

  %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x256xf32>, tensor<1x1x1x1xf32> -> tensor<1x1024x1x256xf32>
  %1 = IE.ReduceMean(%0) {axes_value = [3], keep_dims} : tensor<1x1024x1x256xf32> -> tensor<1x1024x1x1xf32>
  %2 = IE.Add(%1, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1024x1x1xf32>
  %3 = IE.Sqrt(%2) : tensor<1x1024x1x1xf32> -> tensor<1x1024x1x1xf32>
  %4 = IE.Divide(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024x1x256xf32>, tensor<1x1024x1x1xf32> -> tensor<1x1024x1x256xf32>

  %5 = IE.StridedSlice(%4) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 1024, 1, 128], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x1024x1x256xf32> -> tensor<1x1024x1x128xf32>
  %6 = IE.StridedSlice(%4) {begin_mask = [1, 1, 1, 0], begins_attr = [0, 0, 0, 128], ellipsis_mask = [], end_mask = [1, 1, 1, 0], ends_attr = [1, 1024, 1, 256], new_axis_mask = [], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1]} : tensor<1x1024x1x256xf32> -> tensor<1x1024x1x128xf32>

  return %5, %6 : tensor<1x1024x1x128xf32>, tensor<1x1024x1x128xf32>

  // CHECK: [[CST:%.+]] = const.Declare tensor<256xf32> = dense<1.000000e+00> : tensor<256xf32>
  // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[CST]]) {epsilon = 1.000000e+00 : f64} : tensor<1x1024x1x256xf32>, tensor<256xf32> -> tensor<1x1024x1x256xf32>

  // CHECK: [[USER_0:%.+]] = IE.StridedSlice([[RMS]])
  // CHECK: [[USER_1:%.+]] = IE.StridedSlice([[RMS]])

  // CHECK: return [[USER_0]], [[USER_1]] : tensor<1x1024x1x128xf32>, tensor<1x1024x1x128xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumModelF2
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x40x64xf32>, [[ARG1:%.+]]: tensor<1x1x40x64xf32>, [[ARG2:%.+]]: tensor<1x100x40x64xf32>)
func.func @FuseRMSNormForReduceSumModelF2(%arg0: tensor<1x1x40x64xf32>, %arg1: tensor<1x1x40x64xf32>, %arg2: tensor<1x100x40x64xf32>) -> tensor<1x101x40x64xf32> {
  %cst_73 = const.Declare tensor<1x1x1x1xf32> = dense<-52.0> : tensor<1x1x1x1xf32>
  %cst_74 = const.Declare tensor<1x1x1x1xf32> = dense<51.0> : tensor<1x1x1x1xf32>
  %cst_76 = const.Declare tensor<1x1x1x1xf32> = dense<-0.4> : tensor<1x1x1x1xf32>
  %cst_77 = const.Declare tensor<1x1x1x1xf32> = dense<0.5> : tensor<1x1x1x1xf32>

  %cst_75 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  %cst_133 = const.Declare tensor<1x1x1x1xf32> = dense<-131.0> : tensor<1x1x1x1xf32>
  %cst_134 = const.Declare tensor<1x1x1x1xf32> = dense<130.0> : tensor<1x1x1x1xf32>
  %cst_135 = const.Declare tensor<1x1x1x1xf32> = dense<-0.45> : tensor<1x1x1x1xf32>
  %cst_136 = const.Declare tensor<1x1x1x1xf32> = dense<0.55> : tensor<1x1x1x1xf32>
  %cst_137 = const.Declare tensor<1x1x1x1xf32> = dense<-0.6> : tensor<1x1x1x1xf32>
  %cst_138 = const.Declare tensor<1x1x1x1xf32> = dense<0.7> : tensor<1x1x1x1xf32>

  %166 = IE.FakeQuantize(%arg0, %cst_73, %cst_74, %cst_73, %cst_74) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %167 = IE.Power(%166, %cst_75) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %168 = IE.ReduceSum(%167) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %169 = IE.Sqrt(%168) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %170 = IE.Divide(%166, %169) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %171 = IE.FakeQuantize(%170, %cst_76, %cst_77, %cst_76, %cst_77) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>

  %238 = IE.FakeQuantize(%arg1, %cst_133, %cst_134, %cst_133, %cst_134) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %239 = IE.Power(%238, %cst_75) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %240 = IE.ReduceSum(%239) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %241 = IE.Sqrt(%240) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %242 = IE.Divide(%238, %241) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %243 = IE.FakeQuantize(%242, %cst_135, %cst_136, %cst_135, %cst_136) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %244 = IE.Concat(%arg2, %243) {static_offsets = [[0, 0, 0, 0], [0, 100, 0, 0]]} : tensor<1x100x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x101x40x64xf32>
  %245 = IE.FakeQuantize(%244, %cst_137, %cst_138, %cst_137, %cst_138) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x101x40x64xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x101x40x64xf32>

  %246 = IE.Multiply(%171, %245) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32> -> tensor<1x101x40x64xf32>
  return %246 : tensor<1x101x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK-DAG: const.Declare tensor<1x1x1x1xf32>
  // CHECK:  [[FQ1:%.+]] = IE.FakeQuantize([[ARG0]]
  // CHECK:  [[RMS1:%.+]] = IE.RMS([[FQ1]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[FQ2:%.+]] = IE.FakeQuantize([[RMS1]]
  // CHECK:  [[FQ3:%.+]] = IE.FakeQuantize([[ARG1]]
  // CHECK:  [[RMS2:%.+]] = IE.RMS([[FQ3]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[FQ4:%.+]] = IE.FakeQuantize([[RMS2]]
  // CHECK:  [[CONCAT:%.+]] = IE.Concat([[ARG2]], [[FQ4]])
  // CHECK:  [[FQ5:%.+]] = IE.FakeQuantize([[CONCAT]]
  // CHECK:  [[OUT:%.+]] = IE.Multiply([[FQ2]], [[FQ5]])
  // CHECK:  return [[OUT]] : tensor<1x101x40x64xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumDivideHasMultipleUsers
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x40x64xf32>, [[ARG1:%.+]]: tensor<1x1x40x64xf32>, [[ARG2:%.+]]: tensor<1x100x40x64xf32>)
func.func @FuseRMSNormForReduceSumDivideHasMultipleUsers(%arg0: tensor<1x1x40x64xf32>, %arg1: tensor<1x1x40x64xf32>, %arg2: tensor<1x100x40x64xf32>) -> (tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>) {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>

  %167 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %168 = IE.ReduceSum(%167) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %169 = IE.Sqrt(%168) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %170 = IE.Divide(%arg0, %169) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>

  %239 = IE.Power(%arg1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %240 = IE.ReduceSum(%239) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %241 = IE.Sqrt(%240) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %242 = IE.Divide(%arg1, %241) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %244 = IE.Concat(%arg2, %242) {static_offsets = [[0, 0, 0, 0], [0, 100, 0, 0]]} : tensor<1x100x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x101x40x64xf32>

  %246 = IE.Multiply(%170, %244) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32> -> tensor<1x101x40x64xf32>
  return %242, %246 : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK:  [[RMS1:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[RMS2:%.+]] = IE.RMS([[ARG1]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[CONCAT:%.+]] = IE.Concat([[ARG2]], [[RMS2]])
  // CHECK:  [[OUT:%.+]] = IE.Multiply([[RMS1]], [[CONCAT]])
  // CHECK:  return [[RMS2]], [[OUT]] : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumWithAddFixSmallEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x40x64xf32>, [[ARG1:%.+]]: tensor<1x1x40x64xf32>, [[ARG2:%.+]]: tensor<1x100x40x64xf32>)
func.func @FuseRMSNormForReduceSumWithAddFixSmallEpsilon(%arg0: tensor<1x1x40x64xf32>, %arg1: tensor<1x1x40x64xf32>, %arg2: tensor<1x100x40x64xf32>) -> (tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>) {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  %epsilon_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-12> : tensor<1x1x1x1xf32>
  %epsilon_1 = const.Declare tensor<1x1x1x1xf32> = dense<3.700000e-07> : tensor<1x1x1x1xf32>

  %167 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %168 = IE.ReduceSum(%167) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %171 = IE.Add(%168, %epsilon_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x1xf32>
  %169 = IE.Sqrt(%171) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %170 = IE.Divide(%arg0, %169) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>

  %239 = IE.Power(%arg1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %240 = IE.ReduceSum(%239) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %245 = IE.Add(%240, %epsilon_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x1xf32>
  %241 = IE.Sqrt(%245) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %242 = IE.Divide(%arg1, %241) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %244 = IE.Concat(%arg2, %242) {static_offsets = [[0, 0, 0, 0], [0, 100, 0, 0]]} : tensor<1x100x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x101x40x64xf32>

  %246 = IE.Multiply(%170, %244) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32> -> tensor<1x101x40x64xf32>
  return %242, %246 : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK:  [[RMS1:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[RMS2:%.+]] = IE.RMS([[ARG1]], [[GAMMA]]) {epsilon = 3.700000092976552E-7 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[CONCAT:%.+]] = IE.Concat([[ARG2]], [[RMS2]])
  // CHECK:  [[OUT:%.+]] = IE.Multiply([[RMS1]], [[CONCAT]])
  // CHECK:  return [[RMS2]], [[OUT]] : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumWithMaximumFixSmallEpsilon
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x40x64xf32>, [[ARG1:%.+]]: tensor<1x1x40x64xf32>, [[ARG2:%.+]]: tensor<1x100x40x64xf32>)
func.func @FuseRMSNormForReduceSumWithMaximumFixSmallEpsilon(%arg0: tensor<1x1x40x64xf32>, %arg1: tensor<1x1x40x64xf32>, %arg2: tensor<1x100x40x64xf32>) -> (tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>) {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>
  %epsilon_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-12> : tensor<1x1x1x1xf32>
  %epsilon_1 = const.Declare tensor<1x1x1x1xf32> = dense<3.700000e-07> : tensor<1x1x1x1xf32>

  %167 = IE.Power(%arg0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %168 = IE.ReduceSum(%167) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %171 = IE.Maximum(%168, %epsilon_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x1xf32>
  %169 = IE.Sqrt(%171) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %170 = IE.Divide(%arg0, %169) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>

  %239 = IE.Power(%arg1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  %240 = IE.ReduceSum(%239) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %245 = IE.Maximum(%240, %epsilon_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x1xf32>
  %241 = IE.Sqrt(%245) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %242 = IE.Divide(%arg1, %241) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %244 = IE.Concat(%arg2, %242) {static_offsets = [[0, 0, 0, 0], [0, 100, 0, 0]]} : tensor<1x100x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x101x40x64xf32>

  %246 = IE.Multiply(%170, %244) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32> -> tensor<1x101x40x64xf32>
  return %242, %246 : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK:  [[RMS1:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[RMS2:%.+]] = IE.RMS([[ARG1]], [[GAMMA]]) {epsilon = 3.700000092976552E-7 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[CONCAT:%.+]] = IE.Concat([[ARG2]], [[RMS2]])
  // CHECK:  [[OUT:%.+]] = IE.Multiply([[RMS1]], [[CONCAT]])
  // CHECK:  return [[RMS2]], [[OUT]] : tensor<1x1x40x64xf32>, tensor<1x101x40x64xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumMultiplyHasMultipleUsers
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<40x64xf32>)
func.func @FuseRMSNormForReduceSumMultiplyHasMultipleUsers(%arg0: tensor<40x64xf32>) -> (tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>) {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.500000e+00> : tensor<1x1x1x1xf32>
  %epsilon_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e-12> : tensor<1x1x1x1xf32>

  %55 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 40, 64]} : tensor<40x64xf32> -> tensor<1x1x40x64xf32>
  %56 = IE.Multiply(%55, %55) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x1x40x64xf32>
  %57 = IE.ReduceSum(%56) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %58 = IE.Maximum(%57, %epsilon_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x1xf32>
  %59 = IE.Sqrt(%58) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %60 = IE.Divide(%cst_0, %59) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %61 = IE.Multiply(%55, %60) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %62 = IE.Add(%61, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>

  return %62, %61 : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
  // CHECK:  [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK:  [[RMS:%.+]] = IE.RMS([[RESHAPE]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[ADD:%.+]] = IE.Add([[RMS]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  return [[ADD]], [[RMS]] : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>
}

// -----

// CHECK-LABEL: @FuseRMSNormForReduceSumWithClamp
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<40x64xf32>)
func.func @FuseRMSNormForReduceSumWithClamp(%arg0: tensor<40x64xf32>) -> (tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>) {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
  %cst_1 = const.Declare tensor<1x1x1x1xf32> = dense<0.500000e+00> : tensor<1x1x1x1xf32>

  %55 = IE.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 40, 64]} : tensor<40x64xf32> -> tensor<1x1x40x64xf32>
  %56 = IE.Multiply(%55, %55) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x1x40x64xf32>
  %57 = IE.ReduceSum(%56) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %58 = IE.Clamp(%57) {max = 6.550400e+04 : f64, min = 9.999999960041972E-13 : f64} : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %59 = IE.Sqrt(%58) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %60 = IE.Divide(%cst_0, %59) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %61 = IE.Multiply(%55, %60) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>
  %62 = IE.Add(%61, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>

  return %62, %61 : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>

  // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<64xf32> = dense<1.250000e-01> : tensor<64xf32>
  // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e-01> : tensor<1x1x1x1xf32>
  // CHECK:  [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
  // CHECK:  [[RMS:%.+]] = IE.RMS([[RESHAPE]], [[GAMMA]]) {epsilon = 9.9999997171806853E-10 : f64} : tensor<1x1x40x64xf32>, tensor<64xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  [[ADD:%.+]] = IE.Add([[RMS]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x40x64xf32>
  // CHECK:  return [[ADD]], [[RMS]] : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32>
}

// -----

// CHECK-LABEL: @SkipRMSFuseDueToClamp
func.func @SkipRMSFuseDueToClamp(%arg0: tensor<1x1x40x64xf32>) -> tensor<1x1x40x64xf32> {
  %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

  %56 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x64xf32> -> tensor<1x1x40x64xf32>
  %57 = IE.ReduceSum(%56) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf32> -> tensor<1x1x40x1xf32>
  %58 = IE.Clamp(%57) {max = 6.0 : f64, min = 0.0 : f64} : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %59 = IE.Sqrt(%58) : tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %60 = IE.Divide(%cst_0, %59) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x1xf32>
  %61 = IE.Multiply(%arg0, %60) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf32>, tensor<1x1x40x1xf32> -> tensor<1x1x40x64xf32>

  return %61 : tensor<1x1x40x64xf32>

  // CHECK-NOT:  IE.RMS
}
