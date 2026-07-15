//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --fuse-ops-to-matmul="convert-cumsum-to-matmul=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// -----

// CHECK-LABEL: @ConvertCumSum5DAxisMinus2
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x4x256x256xf16>
func.func @ConvertCumSum5DAxisMinus2(%arg0: tensor<1x64x4x256x256xf16>) -> tensor<1x64x4x256x256xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 3 : i64} : tensor<1x64x4x256x256xf16> -> tensor<1x64x4x256x256xf16>
    return %0 : tensor<1x64x4x256x256xf16>

    // batchSize=1*64*4=256, trailingSize=256, L=256 → M=batchSize*trailingSize=65536
    // CHECK:       [[T_IN:%.+]] = IE.Transpose([[INPUT]])
    // CHECK:       [[R1:%.+]] = IE.Reshape([[T_IN]]) {shape_value = [1, 65536, 256]} :
    // CHECK-SAME:      tensor<1x64x4x256x256xf16> -> tensor<1x65536x256xf16>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<256x256xf16>
    // CHECK:       [[MM:%.+]] = IE.MatMul([[R1]], [[CST]]) :
    // CHECK-SAME:      tensor<1x65536x256xf16>, tensor<256x256xf16> -> tensor<1x65536x256xf16>
    // CHECK:       [[R2:%.+]] = IE.Reshape([[MM]]) {shape_value = [1, 64, 4, 256, 256]} :
    // CHECK-SAME:      tensor<1x65536x256xf16> -> tensor<1x64x4x256x256xf16>
    // CHECK:       [[T_OUT:%.+]] = IE.Transpose([[R2]])
    // CHECK:       return [[T_OUT]] : tensor<1x64x4x256x256xf16>
}

// -----

// CHECK-LABEL: @ConvertCumSum5DExclusive
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x4x256x256xf16>
func.func @ConvertCumSum5DExclusive(%arg0: tensor<1x64x4x256x256xf16>) -> tensor<1x64x4x256x256xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 3 : i64, exclusive} : tensor<1x64x4x256x256xf16> -> tensor<1x64x4x256x256xf16>
    return %0 : tensor<1x64x4x256x256xf16>

    // exclusive=true → strict upper-triangular weight (diagonal is 0)
    // CHECK:       IE.Transpose([[INPUT]])
    // CHECK:       IE.Reshape
    // CHECK:       const.Declare tensor<256x256xf16>
    // CHECK:       IE.MatMul
    // CHECK:       [[T_OUT:%.+]] = IE.Transpose
    // CHECK:       return [[T_OUT]] : tensor<1x64x4x256x256xf16>
}

// -----

// CHECK-LABEL: @ConvertCumSum5DReverse
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x4x256x256xf16>
func.func @ConvertCumSum5DReverse(%arg0: tensor<1x64x4x256x256xf16>) -> tensor<1x64x4x256x256xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 3 : i64, reverse} : tensor<1x64x4x256x256xf16> -> tensor<1x64x4x256x256xf16>
    return %0 : tensor<1x64x4x256x256xf16>

    // reverse=true → lower-triangular weight
    // CHECK:       IE.Transpose([[INPUT]])
    // CHECK:       IE.Reshape
    // CHECK:       const.Declare tensor<256x256xf16>
    // CHECK:       IE.MatMul
    // CHECK:       [[T_OUT:%.+]] = IE.Transpose
    // CHECK:       return [[T_OUT]] : tensor<1x64x4x256x256xf16>
}

// -----

// Small axis (total=1*4=4 < MIN_CUMSUM_MATMUL_ELEMENTS) — should NOT be converted.
// CHECK-LABEL: @NoCumSumConversionSmall
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x4xf16>
func.func @NoCumSumConversionSmall(%arg0: tensor<1x4xf16>) -> tensor<1x4xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 1 : i64} : tensor<1x4xf16> -> tensor<1x4xf16>
    return %0 : tensor<1x4xf16>

    // CHECK:       IE.CumSum
    // CHECK-NOT:   IE.MatMul
}

// -----

// 2D input: [256, 128] axis=0, L=256, trailing=128.
// CHECK-LABEL: @ConvertCumSum2D
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x128xf16>
func.func @ConvertCumSum2D(%arg0: tensor<256x128xf16>) -> tensor<256x128xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 0 : i64} : tensor<256x128xf16> -> tensor<256x128xf16>
    return %0 : tensor<256x128xf16>

    // batch=1, L=256, trailing=128 → axis=0, move to last → Transpose([1,0])
    // CHECK:       [[T_IN:%.+]] = IE.Transpose([[INPUT]])
    // CHECK:       [[R1:%.+]] = IE.Reshape([[T_IN]]) {shape_value = [1, 128, 256]} :
    // CHECK-SAME:      tensor<128x256xf16> -> tensor<1x128x256xf16>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<256x256xf16>
    // CHECK:       [[MM:%.+]] = IE.MatMul([[R1]], [[CST]]) :
    // CHECK-SAME:      tensor<1x128x256xf16>, tensor<256x256xf16> -> tensor<1x128x256xf16>
    // CHECK:       IE.Reshape([[MM]])
    // CHECK:       IE.Transpose
    // CHECK:       return
}

// -----

// L=257 > MAX_CUMSUM_MATMUL_L=256 — should NOT be converted.
// CHECK-LABEL: @NoCumSumConversionLargeL
// CHECK-SAME:      [[INPUT:%.+]]: tensor<257x128xf16>
func.func @NoCumSumConversionLargeL(%arg0: tensor<257x128xf16>) -> tensor<257x128xf16> {
    %0 = IE.CumSum(%arg0) {axis_value = 0 : i64} : tensor<257x128xf16> -> tensor<257x128xf16>
    return %0 : tensor<257x128xf16>

    // CHECK:       IE.CumSum
    // CHECK-NOT:   IE.MatMul
}

// -----

// CHECK-LABEL: @ConvertCumSum2DFP32
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x128xf32>
func.func @ConvertCumSum2DFP32(%arg0: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %0 = IE.CumSum(%arg0) {axis_value = 0 : i64} : tensor<256x128xf32> -> tensor<256x128xf32>
    return %0 : tensor<256x128xf32>

    // CHECK:       [[T_IN:%.+]] = IE.Transpose([[INPUT]])
    // CHECK:       [[R1:%.+]] = IE.Reshape([[T_IN]]) {shape_value = [1, 128, 256]} :
    // CHECK-SAME:      tensor<128x256xf32> -> tensor<1x128x256xf32>
    // CHECK:       [[CST:%.+]] = const.Declare tensor<256x256xf32>
    // CHECK:       [[MM:%.+]] = IE.MatMul([[R1]], [[CST]]) :
    // CHECK-SAME:      tensor<1x128x256xf32>, tensor<256x256xf32> -> tensor<1x128x256xf32>
    // CHECK:       IE.Reshape([[MM]])
    // CHECK:       IE.Transpose
    // CHECK:       return
}
