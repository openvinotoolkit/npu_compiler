//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --decompose-mvn="force-decompose=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

// Positive case: NHWC f16 tensor with aligned channels and size above the
// isLargeEnoughForDPUOverSHAVE threshold.  The tensor fits in CMX so the
// regular CMX-overflow path does not fire; forceDecompose=true triggers
// decomposition and sets MVN1NormalizeOp.from_force_decompose = true.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ForceDecomposeMVNNHWCLargeF16
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @ForceDecomposeMVNNHWCLargeF16(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x64x64xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
            : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>
    return %0 : tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK-NOT:  VPU.MVN(
    // CHECK:      [[SC1:%.+]] = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs([[INPUT]] : tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    // CHECK:      [[SUM:%.+]] = VPU.MVN1SumOp([[SC1]]) {across_channels = false, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:     : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>
    // CHECK:      [[MEANVAR:%.+]] = VPU.MVN1MeanVar([[SUM]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16}
    // CHECK-SAME:     : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>
    // CHECK:      [[NORM:%.+]] = VPU.MVN1Normalize([[SC1]], [[MEANVAR]]) {across_channels = false, from_force_decompose = true, normalize_variance = true}
    // CHECK-SAME:     : tensor<1x512x4096x1xf16, {order = #NHWC}>, tensor<1x512x1x2xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    // CHECK:      [[SC2:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs([[NORM]] : tensor<1x512x4096x1xf16, {order = #NHWC}>) -> tensor<1x512x64x64xf16, {order = #NHWC}>
    // CHECK:      return [[SC2]]
}

// -----

// Negative: NCHW layout — forceDecompose requires NHWC; op is left unchanged.

// CHECK-LABEL: func.func @NotForceDecomposeNCHW
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x512x64x64xf16>
func.func @NotForceDecomposeNCHW(%arg0: tensor<1x512x64x64xf16>) -> tensor<1x512x64x64xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
            : tensor<1x512x64x64xf16> -> tensor<1x512x64x64xf16>
    return %0 : tensor<1x512x64x64xf16>

    // CHECK:      VPU.MVN([[INPUT]])
    // CHECK-NOT:  VPU.MVN1SumOp
}

// -----

// Negative: tensor too small — fails isLargeEnoughForDPUOverSHAVE.
// 1x512x4x4xf16 = 16 384 bytes, well below ceil(CMX / numTiles) for NPU5010.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotForceDecomposeSmallTensor
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x512x4x4xf16, {order = #NHWC}>
func.func @NotForceDecomposeSmallTensor(%arg0: tensor<1x512x4x4xf16, {order = #NHWC}>) -> tensor<1x512x4x4xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
            : tensor<1x512x4x4xf16, {order = #NHWC}> -> tensor<1x512x4x4xf16, {order = #NHWC}>
    return %0 : tensor<1x512x4x4xf16, {order = #NHWC}>

    // CHECK:      VPU.MVN([[INPUT]])
    // CHECK-NOT:  VPU.MVN1SumOp
}

// -----

// Negative: channel count not divisible by VPU_CHANNEL_ALIGNMENT (16) —
// NCE.MaxPool requires C % 16 == 0; 513 % 16 = 1 so op is left unchanged.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotForceDecomposeUnalignedChannels
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x513x64x64xf16, {order = #NHWC}>
func.func @NotForceDecomposeUnalignedChannels(%arg0: tensor<1x513x64x64xf16, {order = #NHWC}>) -> tensor<1x513x64x64xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
            : tensor<1x513x64x64xf16, {order = #NHWC}> -> tensor<1x513x64x64xf16, {order = #NHWC}>
    return %0 : tensor<1x513x64x64xf16, {order = #NHWC}>

    // CHECK:      VPU.MVN([[INPUT]])
    // CHECK-NOT:  VPU.MVN1SumOp
}

// -----

// Negative: batch != 1 — RunMVNNormalizeOnDPU expects per-batch MVN statistics but
// extracts mean/scale assuming N==1; forceDecompose must be skipped.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotForceDecomposeBatchGreaterThanOne
// CHECK-SAME:  [[INPUT:%.+]]: tensor<32x512x64x64xf16, {order = #NHWC}>
func.func @NotForceDecomposeBatchGreaterThanOne(%arg0: tensor<32x512x64x64xf16, {order = #NHWC}>) -> tensor<32x512x64x64xf16, {order = #NHWC}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
            : tensor<32x512x64x64xf16, {order = #NHWC}> -> tensor<32x512x64x64xf16, {order = #NHWC}>
    return %0 : tensor<32x512x64x64xf16, {order = #NHWC}>

    // CHECK:      VPU.MVN([[INPUT]])
    // CHECK-NOT:  VPU.MVN1SumOp
}
