//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// A float -> int -> float chain is lossy: both conversions must be preserved.
// CHECK-LABEL: @FloatToIntToFloat
// CHECK: IE.Convert
// CHECK: IE.Convert
func.func @FloatToIntToFloat(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = i64} : tensor<1x3x8x8xf32> -> tensor<1x3x8x8xi64>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x3x8x8xi64> -> tensor<1x3x8x8xf32>
    return %1 : tensor<1x3x8x8xf32>
}

// A f32 -> f16 -> f32 chain is lossy as well: the rounding of the inner conversion must be preserved.
// CHECK-LABEL: @FloatToF16ToFloat
// CHECK: IE.Convert
// CHECK: IE.Convert
func.func @FloatToF16ToFloat(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x8x8xf32> -> tensor<1x3x8x8xf16>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x3x8x8xf16> -> tensor<1x3x8x8xf32>
    return %1 : tensor<1x3x8x8xf32>
}

// The inner conversion targets the input type: it is redundant and the chain fuses into one convert.
// CHECK-LABEL: @RedundantInnerConvert
// CHECK: IE.Convert
// CHECK-NOT: IE.Convert
func.func @RedundantInnerConvert(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x3x8x8xf32> -> tensor<1x3x8x8xf32>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x3x8x8xf32> -> tensor<1x3x8x8xf16>
    return %1 : tensor<1x3x8x8xf16>
}

// The outer conversion targets the intermediate type: also redundant, the chain fuses into one convert.
// CHECK-LABEL: @RedundantOuterConvert
// CHECK: IE.Convert
// CHECK-NOT: IE.Convert
func.func @RedundantOuterConvert(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x3x8x8xf32> -> tensor<1x3x8x8xf16>
    %1 = IE.Convert(%0) {dstElemType = f16} : tensor<1x3x8x8xf16> -> tensor<1x3x8x8xf16>
    return %1 : tensor<1x3x8x8xf16>
}
