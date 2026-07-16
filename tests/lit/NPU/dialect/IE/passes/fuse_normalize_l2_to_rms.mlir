//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-normalize-l2-to-rms --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseNormalizeL2PlusMultiplyToRMS
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf32>)
func.func @FuseNormalizeL2PlusMultiplyToRMS(%arg0: tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32> {
    %cst = const.Declare tensor<1x1x3072xf32> = dense<2.0> : tensor<1x1x3072xf32>
    %0 = IE.NormalizeL2(%arg0) {axes_value = [2], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    return %1 : tensor<1x1x3072xf32>

    // CHECK-NOT: IE.NormalizeL2
    // CHECK-DAG: [[GAMMA:%.+]] = const.Declare tensor<3072xf32>
    // CHECK: [[RMS:%.+]] = IE.RMS([[ARG0]], [[GAMMA]]) {eps = {{.+}} : f64} : tensor<1x1x3072xf32>, tensor<3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: return [[RMS]]
}

// -----

// Any intermediary op between NormalizeL2 and Multiply blocks fusion (fusion requires direct consumer).

// CHECK-LABEL: @NoFuseWhenIntermediaryOpBetweenNormalizeL2AndMultiply
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf32>)
func.func @NoFuseWhenIntermediaryOpBetweenNormalizeL2AndMultiply(%arg0: tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32> {
    %cst_scale = const.Declare tensor<1x1x3072xf32> = dense<2.0> : tensor<1x1x3072xf32>
    %cst_fq_lo = const.Declare tensor<1xf32> = dense<-1.0> : tensor<1xf32>
    %cst_fq_hi = const.Declare tensor<1xf32> = dense<1.0> : tensor<1xf32>
    %0 = IE.NormalizeL2(%arg0) {axes_value = [2], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %1 = IE.FakeQuantize(%0, %cst_fq_lo, %cst_fq_hi, %cst_fq_lo, %cst_fq_hi) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x3072xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32> -> tensor<1x1x3072xf32>
    %2 = IE.Multiply(%1, %cst_scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    return %2 : tensor<1x1x3072xf32>

    // CHECK: [[NORM:%.+]] = IE.NormalizeL2([[ARG0]])
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[NORM]],
    // CHECK-NOT: IE.RMS
    // CHECK: [[MUL:%.+]] = IE.Multiply([[FQ]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: return [[MUL]] : tensor<1x1x3072xf32>
}

// -----

// NormalizeL2 with multiple uses must not be fused.

// CHECK-LABEL: @NoFuseWhenNormalizeL2HasMultipleUses
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x1x3072xf32>)
func.func @NoFuseWhenNormalizeL2HasMultipleUses(%arg0: tensor<1x1x3072xf32>) -> (tensor<1x1x3072xf32>, tensor<1x1x3072xf32>) {
    %cst = const.Declare tensor<1x1x3072xf32> = dense<2.0> : tensor<1x1x3072xf32>
    %0 = IE.NormalizeL2(%arg0) {axes_value = [2], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    return %0, %1 : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>

    // CHECK: [[NORM:%.+]] = IE.NormalizeL2([[ARG0]])
    // CHECK-NOT: IE.RMS
    // CHECK: [[MUL:%.+]] = IE.Multiply([[NORM]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf32>, tensor<1x1x3072xf32> -> tensor<1x1x3072xf32>
    // CHECK: return [[NORM]], [[MUL]] : tensor<1x1x3072xf32>, tensor<1x1x3072xf32>
}

// -----

// NormalizeL2 reducing on a non-innermost axis must not be fused.

// CHECK-LABEL: @NoFuseWhenNotInnermostAxis
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x3072x64xf32>)
func.func @NoFuseWhenNotInnermostAxis(%arg0: tensor<1x3072x64xf32>) -> tensor<1x3072x64xf32> {
    %cst = const.Declare tensor<1x3072x64xf32> = dense<2.0> : tensor<1x3072x64xf32>
    %0 = IE.NormalizeL2(%arg0) {axes_value = [1], eps = 1.000000e-05 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x3072x64xf32> -> tensor<1x3072x64xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072x64xf32>, tensor<1x3072x64xf32> -> tensor<1x3072x64xf32>
    return %1 : tensor<1x3072x64xf32>

    // CHECK: [[NORM:%.+]] = IE.NormalizeL2([[ARG0]])
    // CHECK-NOT: IE.RMS
    // CHECK: [[MUL:%.+]] = IE.Multiply([[NORM]], {{%.+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3072x64xf32>, tensor<1x3072x64xf32> -> tensor<1x3072x64xf32>
    // CHECK: return [[MUL]] : tensor<1x3072x64xf32>
}
