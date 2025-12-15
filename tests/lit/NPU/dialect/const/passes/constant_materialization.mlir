//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Note: these tests verify that constant materialization routine works
// correctly on top of our own custom constant infrastructure.

// CHECK-LABEL: @OptimizeDummyAdd
// CHECK-SAME: ([[IN:%.+]]: tensor<15xf16>)
func.func @OptimizeDummyAdd(%input : tensor<15xf16>) -> tensor<15xf16> {
  %cst = const.Declare tensor<1xf16> = dense<0.0> : tensor<1xf16>
  %res = IE.Add(%input, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<15xf16>, tensor<1xf16> -> tensor<15xf16>
  return %res : tensor<15xf16>

  // CHECK: return [[IN]]
}

// -----

// CHECK-LABEL: @OptimizeDummyMultiply
// CHECK-SAME: ([[IN:%.+]]: tensor<15xf16>)
func.func @OptimizeDummyMultiply(%input : tensor<15xf16>) -> tensor<15xf16> {
  %cst = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
  %res = IE.Multiply(%input, %cst) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<15xf16>, tensor<1xf16> -> tensor<15xf16>
  return %res : tensor<15xf16>

  // CHECK: return [[IN]]
}

// -----

// CHECK-LABEL: @OptimizeDummySoftmaxTwice
// CHECK-SAME: ([[IN:%.+]]: tensor<1x15xf16>)
func.func @OptimizeDummySoftmaxTwice(%input : tensor<1x15xf16>)
    -> (tensor<1x15xf16>, tensor<1x15xf16>) {
  %res1 = IE.SoftMax(%input) { axisInd = 0 } : tensor<1x15xf16> -> tensor<1x15xf16>
  %res2 = IE.SoftMax(%input) { axisInd = 0 } : tensor<1x15xf16> -> tensor<1x15xf16>
  return %res1, %res2 : tensor<1x15xf16>, tensor<1x15xf16>

  // CHECK: [[NEW_CST:%.+]] = const.Declare tensor<1x15xf16> = dense<1.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f16>]
  // CHECK: return [[NEW_CST]], [[NEW_CST]]
}

// -----

// CHECK-LABEL: @MaterializeConvert
func.func @MaterializeConvert() -> tensor<1xf16> {
  %cst = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
  %res = IE.Convert(%cst) {dstElemType = f16} : tensor<1xf32> -> tensor<1xf16>
  return %res : tensor<1xf16>

  // CHECK: [[NEW_CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f16>]
  // CHECK: return [[NEW_CST]]
}

// -----

// CHECK-LABEL: @MaterializeTwoConstantsFromNonConst
// CHECK-SAME: ([[IN:%.+]]: tensor<1x15xf16>)
func.func @MaterializeTwoConstantsFromNonConst(%input : tensor<1x15xf16>)
    -> (tensor<1x15xf32>, tensor<1x15xf16>) {
  %noop = IE.SoftMax(%input) { axisInd = 0 } : tensor<1x15xf16> -> tensor<1x15xf16>
  %cvt = IE.Convert(%noop) {dstElemType = f32} : tensor<1x15xf16> -> tensor<1x15xf32>
  %add = IE.Add(%noop, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<1x15xf16>, tensor<1x15xf16> -> tensor<1x15xf16>
  return %cvt, %add : tensor<1x15xf32>, tensor<1x15xf16>

  // CHECK: [[NEW_CST1:%.+]] = const.Declare tensor<1x15xf16> = dense<1.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f16>]
  // CHECK: [[NEW_CST2:%.+]] = const.Declare tensor<1x15xf32> = dense<1.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f32>]
  // CHECK: [[ADD:%.+]] = IE.Add([[IN]], [[NEW_CST1]])
  // CHECK: return [[NEW_CST2]], [[ADD]]
}

// -----

// CHECK-LABEL: @MaterializeTwoConstantsFromNonConst2
// CHECK-SAME: ([[IN:%.+]]: tensor<1x15xf16>)
func.func @MaterializeTwoConstantsFromNonConst2(%input : tensor<1x15xf16>)
    -> (tensor<1x15xf32>, tensor<1x15xf16>) {
  %noop = IE.LogSoftmax(%input) { axisInd = 0 } : tensor<1x15xf16> -> tensor<1x15xf16>
  %cvt = IE.Convert(%noop) {dstElemType = f32} : tensor<1x15xf16> -> tensor<1x15xf32>
  %mult = IE.Multiply(%noop, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<1x15xf16>, tensor<1x15xf16> -> tensor<1x15xf16>
  return %cvt, %mult : tensor<1x15xf32>, tensor<1x15xf16>

  // CHECK: [[NEW_CST1:%.+]] = const.Declare tensor<1x15xf16> = dense<0.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f16>]
  // CHECK: [[NEW_CST2:%.+]] = const.Declare tensor<1x15xf32> = dense<0.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f32>]
  // CHECK: [[MULT:%.+]] = IE.Multiply([[IN]], [[NEW_CST1]])
  // CHECK: return [[NEW_CST2]], [[MULT]]
}

// -----

// CHECK-LABEL: @MaterializeTwoConstantsFromOne
// CHECK-SAME: ([[IN:%.+]]: tensor<1x15xf16>)
func.func @MaterializeTwoConstantsFromOne(%input : tensor<1x15xf16>)
    -> (tensor<1x15xf32>, tensor<1x15xf16>) {
  %cst = const.Declare tensor<1x15xf16> = dense<2.0> : tensor<1x15xf16>
  %cvt = IE.Convert(%cst) {dstElemType = f32} : tensor<1x15xf16> -> tensor<1x15xf32>
  %mult = IE.Multiply(%cst, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<1x15xf16>, tensor<1x15xf16> -> tensor<1x15xf16>
  return %cvt, %mult : tensor<1x15xf32>, tensor<1x15xf16>

  // CHECK: [[NEW_CST1:%.+]] = const.Declare tensor<1x15xf16> = dense<2.000000e+00>
  // CHECK: [[NEW_CST2:%.+]] = const.Declare tensor<1x15xf32> = dense<2.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f32>]
  // CHECK: [[MULT:%.+]] = IE.Multiply([[IN]], [[NEW_CST1]])
  // CHECK: return [[NEW_CST2]], [[MULT]]
}

// -----

// CHECK-LABEL: @MaterializeTwoConstantsFromTwo
// CHECK-SAME: ([[IN:%.+]]: tensor<1x15xf16>)
func.func @MaterializeTwoConstantsFromTwo(%input : tensor<1x15xf16>)
    -> (tensor<1x15xf32>, tensor<1x15xf16>) {
  %cst1 = const.Declare tensor<1x15xf16> = dense<2.0> : tensor<1x15xf16>
  %cvt = IE.Convert(%cst1) {dstElemType = f32} : tensor<1x15xf16> -> tensor<1x15xf32>

  %cst2 = const.Declare tensor<1x15xf16> = dense<2.0> : tensor<1x15xf16>
  %mult = IE.Multiply(%cst2, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> }
    : tensor<1x15xf16>, tensor<1x15xf16> -> tensor<1x15xf16>
  return %cvt, %mult : tensor<1x15xf32>, tensor<1x15xf16>

  // CHECK: [[NEW_CST1:%.+]] = const.Declare tensor<1x15xf16> = dense<2.000000e+00>
  // CHECK: [[NEW_CST2:%.+]] = const.Declare tensor<1x15xf32> = dense<2.000000e+00>
  // CHECK-SAME: [#const.CastElemType<f32>]
  // CHECK: [[MULT:%.+]] = IE.Multiply([[IN]], [[NEW_CST1]])
  // CHECK: return [[NEW_CST2]], [[MULT]]
}
