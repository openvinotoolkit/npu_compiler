//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --unroll-reducemin-all-axes %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: @UnrollMultiAxisReduceMinAndBigTensor
func.func @UnrollMultiAxisReduceMinAndBigTensor(%arg0: tensor<1x1x256x512xf16>) -> tensor<1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x1x256x512xf16> -> tensor<1xf16>
  return %0 : tensor<1xf16>

  // CHECK:       [[REDUCEMIN_1:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [3]} : tensor<1x1x256x512xf16> -> tensor<1x1x256xf16>
  // CHECK:       [[REDUCEMIN_2:%.*]] = IE.ReduceMin([[REDUCEMIN_1]]) {
  // CHECK-DAG:       axes_value = [2]} : tensor<1x1x256xf16> -> tensor<1x1xf16>
  // CHECK:       [[REDUCEMIN_3:%.*]] = IE.ReduceMin([[REDUCEMIN_2]]) {
  // CHECK-DAG:       axes_value = [0]} : tensor<1x1xf16> -> tensor<1xf16>
}

// CHECK-LABEL: @UnrollTwoAxisReduceMinAndBigTensor
func.func @UnrollTwoAxisReduceMinAndBigTensor(%arg0: tensor<1x1x256x512xf16>) -> tensor<1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [2, 3]} : tensor<1x1x256x512xf16> -> tensor<1x1xf16>
  return %0 : tensor<1x1xf16>

  // CHECK:       [[REDUCEMIN_1:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [3]} : tensor<1x1x256x512xf16> -> tensor<1x1x256xf16>
  // CHECK:       [[REDUCEMIN_2:%.*]] = IE.ReduceMin([[REDUCEMIN_1]]) {
  // CHECK-DAG:       axes_value = [2]} : tensor<1x1x256xf16> -> tensor<1x1xf16>
}

// CHECK-LABEL: @UnrollMultiAxisReduceMinAndThreeNonTrivialDimBigTensor
func.func @UnrollMultiAxisReduceMinAndThreeNonTrivialDimBigTensor(%arg0: tensor<1x1024x128x256xf16>) -> tensor<1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x1024x128x256xf16> -> tensor<1xf16>
  return %0 : tensor<1xf16>

  // CHECK:       [[REDUCEMIN_1:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [1]} : tensor<1x1024x128x256xf16> -> tensor<1x128x256xf16>
  // CHECK:       [[REDUCEMIN_2:%.*]] = IE.ReduceMin([[REDUCEMIN_1]]) {
  // CHECK-DAG:       axes_value = [2]} : tensor<1x128x256xf16> -> tensor<1x128xf16>
  // CHECK:       [[REDUCEMIN_3:%.*]] = IE.ReduceMin([[REDUCEMIN_2]]) {
  // CHECK-DAG:       axes_value = [1]} : tensor<1x128xf16> -> tensor<1xf16>
}

// CHECK-LABEL: @DoNotUnrollSingleAxisReduceMinAndThreeNonTrivialDimBigTensor
func.func @DoNotUnrollSingleAxisReduceMinAndThreeNonTrivialDimBigTensor(%arg0: tensor<1x1024x128x256xf16>) -> tensor<1x128x256xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [1]} : tensor<1x1024x128x256xf16> -> tensor<1x128x256xf16>
  return %0 : tensor<1x128x256xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [1]} : tensor<1x1024x128x256xf16> -> tensor<1x128x256xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1x128x256xf16>
}

// CHECK-LABEL: @DoNotUnrollMultiAxisReduceMinAndSmallTensor
func.func @DoNotUnrollMultiAxisReduceMinAndSmallTensor(%arg0: tensor<1x1x128x64xf16>) -> tensor<1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x1x128x64xf16> -> tensor<1xf16>
  return %0 : tensor<1xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [0, 1, 2, 3]} : tensor<1x1x128x64xf16> -> tensor<1xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1xf16>
}

// CHECK-LABEL: @DoNotUnrollMultiAxisReduceMinAndSingleNonTrivialDimSmallTensor
func.func @DoNotUnrollMultiAxisReduceMinAndSingleNonTrivialDimSmallTensor(%arg0: tensor<1x1x1x256xf16>) -> tensor<1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2, 3]} : tensor<1x1x1x256xf16> -> tensor<1xf16>
  return %0 : tensor<1xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [0, 1, 2, 3]} : tensor<1x1x1x256xf16> -> tensor<1xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1xf16>
}

// CHECK-LABEL: @DoNotUnrollSingleAxisReduceMinAndSingleNonTrivialDimSmallTensor
func.func @DoNotUnrollSingleAxisReduceMinAndSingleNonTrivialDimSmallTensor(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [3]} : tensor<1x1x1x256xf16> -> tensor<1x1x1xf16>
  return %0 : tensor<1x1x1xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [3]} : tensor<1x1x1x256xf16> -> tensor<1x1x1xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1x1x1xf16>
}

// CHECK-LABEL: @DoNotUnrollTwoAxisReduceMinAndSmallTensor
func.func @DoNotUnrollTwoAxisReduceMinAndSmallTensor(%arg0: tensor<1x1x16x16xf16>) -> tensor<1x1xf16> {
  %0 = IE.ReduceMin(%arg0) {axes_value = [2, 3]} : tensor<1x1x16x16xf16> -> tensor<1x1xf16>
  return %0 : tensor<1x1xf16>

  // CHECK:       [[OUTPUT:%.*]] = IE.ReduceMin([[INPUT:%.*]]) {
  // CHECK-DAG:       axes_value = [2, 3]} : tensor<1x1x16x16xf16> -> tensor<1x1xf16>
  // CHECK:       return [[OUTPUT:%.*]] : tensor<1x1xf16>
}
