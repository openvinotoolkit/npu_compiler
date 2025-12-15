//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-mvn6-scale-bias %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL: @FuseMVN6WithScale
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x150x64x8xf16>, [[SCALE:%.+]]: tensor<1x150x64x8xf16>
func.func @FuseMVN6WithScale(%arg0: tensor<1x150x64x8xf16>, %arg1: tensor<1x150x64x8xf16>) -> tensor<1x150x64x8xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    return %1 : tensor<1x150x64x8xf16>

    // CHECK: [[OUTPUT:%.+]] =  IE.MVN6([[INPUT]], [[SCALE]]) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x150x64x8xf16>
}

// -----

// CHECK-LABEL: @FuseMVN6WithBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x150x64x8xf16>, [[BIAS:%.+]]: tensor<1x150x64x8xf16>
func.func @FuseMVN6WithBias(%arg0: tensor<1x150x64x8xf16>, %arg1: tensor<1x150x64x8xf16>) -> tensor<1x150x64x8xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %1 = IE.Add(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    return %1 : tensor<1x150x64x8xf16>

    // CHECK: [[OUTPUT:%.+]] =  IE.MVN6([[INPUT]], [[BIAS]]) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 1, 0>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x150x64x8xf16>
}

// -----

// CHECK-LABEL: @FuseMVN6WithScaleBias
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x150x64x8xf16>, [[SCALE:%.+]]: tensor<1x150x64x8xf16>, [[BIAS:%.+]]: tensor<1x150x64x8xf16>
  func.func @FuseMVN6WithScaleBias(%arg0: tensor<1x150x64x8xf16>, %arg1: tensor<1x150x64x8xf16>, %arg2: tensor<1x150x64x8xf16>) -> tensor<1x150x64x8xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %2 = IE.Add(%1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    return %2 : tensor<1x150x64x8xf16>

    // CHECK: [[OUTPUT:%.+]] =  IE.MVN6([[INPUT]], [[SCALE]], [[BIAS]]) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1, 0>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x150x64x8xf16>
}

// -----

// CHECK-LABEL: @FuseMVN6WithMulOnly
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x150x64x8xf16>, [[SCALE:%.+]]: tensor<1x150x64x8xf16>
  func.func @FuseMVN6WithMulOnly(%arg0: tensor<1x150x64x8xf16>, %arg1: tensor<1x150x64x8xf16>) -> tensor<1x150x64x8xf16> {
    %0 = IE.MVN6(%arg0) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %1 = IE.Multiply(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %2 = IE.Sigmoid(%1) : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    %3 = IE.Add(%1,%2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    return %3 : tensor<1x150x64x8xf16>

    // CHECK: [[MVN6:%.+]] = IE.MVN6([[INPUT]], [[SCALE]]) {axes_value = [2, 3], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 0, 0>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: [[SIGM:%.+]] = IE.Sigmoid([[MVN6]]) : tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: [[OUT:%.+]]  = IE.Add([[MVN6]], [[SIGM]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x150x64x8xf16>, tensor<1x150x64x8xf16> -> tensor<1x150x64x8xf16>
    // CHECK: return [[OUT]] : tensor<1x150x64x8xf16>
}

// -----

// CHECK-LABEL: @FuseMVN6WithScaleBiasRanks
// CHECK-SAME:    [[INPUT:%.+]]: tensor<151x1x768xf16>, [[SCALE:%.+]]: tensor<768xf16>, [[BIAS:%.+]]: tensor<1x1x768xf16>
  func.func @FuseMVN6WithScaleBiasRanks(%input: tensor<151x1x768xf16>, %scale: tensor<768xf16>, %bias: tensor<1x1x768xf16>) -> tensor<151x1x768xf16> {
    %0 = IE.MVN6(%input) {axes_value = [-1], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>} : tensor<151x1x768xf16> -> tensor<151x1x768xf16>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x768xf16>, tensor<768xf16> -> tensor<151x1x768xf16>
    %2 = IE.Add(%1, %bias) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<151x1x768xf16>, tensor<1x1x768xf16> -> tensor<151x1x768xf16>
    return %2 : tensor<151x1x768xf16>

    // CHECK:       [[OUTPUT:%.+]] =  IE.MVN6([[INPUT]], [[SCALE]], [[BIAS]]) {axes_value = [-1], eps = 5.000000e-07 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1, 0>}
    // CHECK-SAME:                    : tensor<151x1x768xf16>, tensor<768xf16>, tensor<1x1x768xf16> -> tensor<151x1x768xf16>
    // CHECK:       return [[OUTPUT]] : tensor<151x1x768xf16>
}
