//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-softmax --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @FuseSoftmax_DecomposedSoftmaxPattern
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<64x199x63xf32>)
func.func @FuseSoftmax_DecomposedSoftmaxPattern(%arg0: tensor<64x199x63xf32>) -> tensor<64x199x63xf32> {
  %cst_75 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_76 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_77 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_78 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_79 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_80 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_81 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_82 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_83 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_84 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_85 = const.Declare tensor<1x1x1xf32> = dense<-1.280000e+02> : tensor<1x1x1xf32>
  %cst_86 = const.Declare tensor<1x1x1xf32> = dense<1.270000e+02> : tensor<1x1x1xf32>
  %cst_87 = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
  %cst_88 = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
  %cst_89 = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
  %cst_90 = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
  %cst_91 = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
  %cst_92 = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
  %cst_93 = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
  %cst_94 = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>

  %0 = IE.FakeQuantize(%arg0, %cst_75, %cst_76, %cst_77, %cst_78) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<64x199x63xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<64x199x63xf32>
  %1 = IE.ReduceMax(%0) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %2 = IE.FakeQuantize(%1, %cst_79, %cst_80, %cst_81, %cst_82) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<64x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<64x1x1xf32>
  %3 = IE.Subtract(%0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>
  %4 = IE.FakeQuantize(%3, %cst_83, %cst_84, %cst_85, %cst_86) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<64x199x63xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<64x199x63xf32>
  %5 = IE.Exp(%4) : tensor<64x199x63xf32> -> tensor<64x199x63xf32>
  %6 = IE.FakeQuantize(%5, %cst_87, %cst_88, %cst_89, %cst_90) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<64x199x63xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<64x199x63xf32>
  %7 = IE.ReduceSum(%6) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %8 = IE.FakeQuantize(%7, %cst_91, %cst_92, %cst_93, %cst_94) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<64x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<64x1x1xf32>
  %9 = IE.Divide(%6, %8) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>

  return %9 : tensor<64x199x63xf32>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG0]]
    // CHECK: [[RESHAPE_IN:%.+]] = IE.AffineReshape([[FQ]]) {dim_mapping = {{\[\[0, 1\], \[2\], \[2\]\]}}, shape_value = [1, 64, 12537]} : tensor<64x199x63xf32> -> tensor<1x64x12537xf32>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE_IN]]) {axisInd = 2 : i64} : tensor<1x64x12537xf32> -> tensor<1x64x12537xf32>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SOFTMAX]]) {dim_mapping = {{\[\[0\], \[0\], \[1, 2\]\]}}, shape_value = [64, 199, 63]} : tensor<1x64x12537xf32> -> tensor<64x199x63xf32>
    // CHECK: return [[RESHAPE_OUT]] : tensor<64x199x63xf32>
}

// -----

// CHECK-LABEL: @FuseSoftmax_WithoutFakeQuantize
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<64x199x63xf32>)
func.func @FuseSoftmax_WithoutFakeQuantize(%arg0: tensor<64x199x63xf32>) -> tensor<64x199x63xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>
  %2 = IE.Exp(%1) : tensor<64x199x63xf32> -> tensor<64x199x63xf32>
  %3 = IE.ReduceSum(%2) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %4 = IE.Divide(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>

  return %4 : tensor<64x199x63xf32>

    // CHECK: [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG0]]) {dim_mapping = {{\[\[0, 1\], \[2\], \[2\]\]}}, shape_value = [1, 64, 12537]} : tensor<64x199x63xf32> -> tensor<1x64x12537xf32>
    // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE_IN]]) {axisInd = 2 : i64} : tensor<1x64x12537xf32> -> tensor<1x64x12537xf32>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SOFTMAX]]) {dim_mapping = {{\[\[0\], \[0\], \[1, 2\]\]}}, shape_value = [64, 199, 63]} : tensor<1x64x12537xf32> -> tensor<64x199x63xf32>
    // CHECK: return [[RESHAPE_OUT]] : tensor<64x199x63xf32>
}

// -----

// CHECK-LABEL: @FuseSoftmax_TransposeCase
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<2x4x8xf32>)
func.func @FuseSoftmax_TransposeCase(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [0, 2], keep_dims} : tensor<2x4x8xf32> -> tensor<1x4x1xf32>
  %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x8xf32>, tensor<1x4x1xf32> -> tensor<2x4x8xf32>
  %2 = IE.Exp(%1) : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %3 = IE.ReduceSum(%2) {axes_value = [0, 2], keep_dims} : tensor<2x4x8xf32> -> tensor<1x4x1xf32>
  %4 = IE.Divide(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x4x8xf32>, tensor<1x4x1xf32> -> tensor<2x4x8xf32>

  return %4 : tensor<2x4x8xf32>

  // CHECK: [[TRANSPOSE_IN:%.+]] = IE.Transpose([[ARG0]]) {order_value = #HCW} : tensor<2x4x8xf32> -> tensor<4x2x8xf32>
  // CHECK: [[RESHAPE_IN:%.+]] = IE.AffineReshape([[TRANSPOSE_IN]]) {dim_mapping = {{\[\[0, 1\], \[2\], \[2\]\]}}, shape_value = [1, 4, 16]} : tensor<4x2x8xf32> -> tensor<1x4x16xf32>
  // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[RESHAPE_IN]]) {axisInd = 2 : i64} : tensor<1x4x16xf32> -> tensor<1x4x16xf32>
  // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[SOFTMAX]]) {dim_mapping = {{\[\[0\], \[0\], \[1, 2\]\]}}, shape_value = [4, 2, 8]} : tensor<1x4x16xf32> -> tensor<4x2x8xf32>
  // CHECK: [[TRANSPOSE_OUT:%.+]] = IE.Transpose([[RESHAPE_OUT]]) {order_value = #HCW} : tensor<4x2x8xf32> -> tensor<2x4x8xf32>
  // CHECK: return [[TRANSPOSE_OUT]] : tensor<2x4x8xf32>
}

// -----

// CHECK-LABEL: @FuseSoftmax_NoMatch_MissingExp
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<64x199x63xf32>)
func.func @FuseSoftmax_NoMatch_MissingExp(%arg0: tensor<64x199x63xf32>) -> tensor<64x199x63xf32> {
  %0 = IE.ReduceMax(%arg0) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %1 = IE.Subtract(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>
  %2 = IE.ReduceSum(%1) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
  %3 = IE.Divide(%1, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>

  return %3 : tensor<64x199x63xf32>

    // CHECK: [[REDUCEMAX:%.+]] = IE.ReduceMax([[ARG0]]) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
    // CHECK: [[SUBTRACT:%.+]] = IE.Subtract([[ARG0]], [[REDUCEMAX]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>
    // CHECK: [[REDUCESUM:%.+]] = IE.ReduceSum([[SUBTRACT]]) {axes_value = [1, 2], keep_dims} : tensor<64x199x63xf32> -> tensor<64x1x1xf32>
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[SUBTRACT]], [[REDUCESUM]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x199x63xf32>, tensor<64x1x1xf32> -> tensor<64x199x63xf32>
    // CHECK: return [[DIVIDE]] : tensor<64x199x63xf32>
}
