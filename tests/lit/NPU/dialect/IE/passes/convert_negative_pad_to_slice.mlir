//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-negative-pad-to-slice %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-LABEL: @ConvertNegativePadBeginToSlice
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2048x1024xf16>)
func.func @ConvertNegativePadBeginToSlice(%arg0: tensor<1x2048x1024xf16>) -> tensor<1x2048x3xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, -1021], pads_end_attr = [0, 0, 0]}
                        : tensor<1x2048x1024xf16> -> tensor<1x2048x3xf16>
    return %0 : tensor<1x2048x3xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 1021] [1, 2048, 3] : tensor<1x2048x1024xf16> to tensor<1x2048x3xf16>
    // CHECK:       return [[SLICE]] : tensor<1x2048x3xf16>
}

// -----

// CHECK-LABEL: @ConvertNegativePadEndToSlice
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertNegativePadEndToSlice(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x16x10xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 0, 0, -6]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x16x10xf16>
    return %0 : tensor<1x8x16x10xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 0, 0] [1, 8, 16, 10] : tensor<1x8x16x16xf16> to tensor<1x8x16x10xf16>
    // CHECK:       return [[SLICE]] : tensor<1x8x16x10xf16>
}

// -----

// CHECK-LABEL: @ConvertNegativePadBothEndsToSlice
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertNegativePadBothEndsToSlice(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x10x10xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, -3, -2], pads_end_attr = [0, 0, -3, -4]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x10x10xf16>
    return %0 : tensor<1x8x10x10xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 3, 2] [1, 8, 10, 10] : tensor<1x8x16x16xf16> to tensor<1x8x10x10xf16>
    // CHECK:       return [[SLICE]] : tensor<1x8x10x10xf16>
}

// -----

// CHECK-LABEL: @ConvertMixedPadToSliceAndPad
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertMixedPadToSliceAndPad(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x14x18xf16> {
    // pads_begin: H=-2 (slice 2 from begin), W=+2 (pad 2 at begin)
    // pads_end: H=0, W=0
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 0, -2, 2], pads_end_attr = [0, 0, 0, 0]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x14x18xf16>
    return %0 : tensor<1x8x14x18xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 2, 0] [1, 8, 14, 16] : tensor<1x8x16x16xf16> to tensor<1x8x14x16xf16>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[SLICE]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 2], pads_end_attr = [0, 0, 0, 0]}
    // CHECK-SAME:       tensor<1x8x14x16xf16> -> tensor<1x8x14x18xf16>
    // CHECK:       return [[PAD]] : tensor<1x8x14x18xf16>
}

// -----

// CHECK-LABEL: @ConvertMixedPadBeginEndToSliceAndPad
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x20x20xf16>)
func.func @ConvertMixedPadBeginEndToSliceAndPad(%arg0: tensor<1x8x20x20xf16>) -> tensor<1x8x15x22xf16> {
    // H: begin=-3 (slice 3), end=-2 (slice 2) => 20-3-2=15
    // W: begin=+1, end=+1 => 20+1+1=22
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, -3, 1], pads_end_attr = [0, 0, -2, 1]}
                        : tensor<1x8x20x20xf16> -> tensor<1x8x15x22xf16>
    return %0 : tensor<1x8x15x22xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 3, 0] [1, 8, 15, 20] : tensor<1x8x20x20xf16> to tensor<1x8x15x20xf16>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[SLICE]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 1], pads_end_attr = [0, 0, 0, 1]}
    // CHECK-SAME:       tensor<1x8x15x20xf16> -> tensor<1x8x15x22xf16>
    // CHECK:       return [[PAD]] : tensor<1x8x15x22xf16>
}

// -----

// CHECK-LABEL: @ConvertMixedPadMultiAxis
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x16x32x32xf16>)
func.func @ConvertMixedPadMultiAxis(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x18x28x34xf16> {
    // C: begin=+2 (pad), end=0
    // H: begin=-2 (slice), end=-2 (slice) => 32-2-2=28
    // W: begin=+1 (pad), end=+1 (pad) => 32+1+1=34
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 2, -2, 1], pads_end_attr = [0, 0, -2, 1]}
                        : tensor<1x16x32x32xf16> -> tensor<1x18x28x34xf16>
    return %0 : tensor<1x18x28x34xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 2, 0] [1, 16, 28, 32] : tensor<1x16x32x32xf16> to tensor<1x16x28x32xf16>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[SLICE]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 2, 0, 1], pads_end_attr = [0, 0, 0, 1]}
    // CHECK-SAME:       tensor<1x16x28x32xf16> -> tensor<1x18x28x34xf16>
    // CHECK:       return [[PAD]] : tensor<1x18x28x34xf16>
}

// -----

// CHECK-LABEL: @ConvertNegativePadWith3DShape
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x96x1024xf16>)
func.func @ConvertNegativePadWith3DShape(%arg0: tensor<1x96x1024xf16>) -> tensor<1x96x512xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, -256], pads_end_attr = [0, 0, -256]}
                        : tensor<1x96x1024xf16> -> tensor<1x96x512xf16>
    return %0 : tensor<1x96x512xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 256] [1, 96, 512] : tensor<1x96x1024xf16> to tensor<1x96x512xf16>
    // CHECK:       return [[SLICE]] : tensor<1x96x512xf16>
}

// -----

// CHECK-LABEL: @ConvertNegativePadEdgeMode
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertNegativePadEdgeMode(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x10x10xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pads_begin_attr = [0, 0, -3, -2], pads_end_attr = [0, 0, -3, -4]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x10x10xf16>
    return %0 : tensor<1x8x10x10xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 3, 2] [1, 8, 10, 10] : tensor<1x8x16x16xf16> to tensor<1x8x10x10xf16>
    // CHECK:       return [[SLICE]] : tensor<1x8x10x10xf16>
}

// -----

// CHECK-LABEL: @ConvertNegativePadReflectMode
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertNegativePadReflectMode(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x10x10xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pads_begin_attr = [0, 0, -3, -2], pads_end_attr = [0, 0, -3, -4]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x10x10xf16>
    return %0 : tensor<1x8x10x10xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 3, 2] [1, 8, 10, 10] : tensor<1x8x16x16xf16> to tensor<1x8x10x10xf16>
    // CHECK:       return [[SLICE]] : tensor<1x8x10x10xf16>
}

// -----

// CHECK-LABEL: @ConvertMixedPadEdgeMode
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @ConvertMixedPadEdgeMode(%arg0: tensor<1x8x16x16xf16>) -> tensor<1x8x14x18xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pads_begin_attr = [0, 0, -2, 2], pads_end_attr = [0, 0, 0, 0]}
                        : tensor<1x8x16x16xf16> -> tensor<1x8x14x18xf16>
    return %0 : tensor<1x8x14x18xf16>

    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG0]] [0, 0, 2, 0] [1, 8, 14, 16] : tensor<1x8x16x16xf16> to tensor<1x8x14x16xf16>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[SLICE]]) {mode = #IE.pad_mode<EDGE>, pads_begin_attr = [0, 0, 0, 2], pads_end_attr = [0, 0, 0, 0]}
    // CHECK-SAME:       tensor<1x8x14x16xf16> -> tensor<1x8x14x18xf16>
    // CHECK:       return [[PAD]] : tensor<1x8x14x18xf16>
}

// -----

// CHECK-LABEL: @SkipPositivePadOnly
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x8x16x16xf16>)
func.func @SkipPositivePadOnly(%arg0: tensor<1x8x16x16xf16>) -> tensor<4x8x16x16xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [1, 0, 0, 0], pads_end_attr = [2, 0, 0, 0]}
                        : tensor<1x8x16x16xf16> -> tensor<4x8x16x16xf16>
    return %0 : tensor<4x8x16x16xf16>

    // CHECK:       [[PAD:%.+]] = IE.Pad([[ARG0]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [1, 0, 0, 0], pads_end_attr = [2, 0, 0, 0]}
    // CHECK-SAME:       tensor<1x8x16x16xf16> -> tensor<4x8x16x16xf16>
    // CHECK:       return [[PAD]] : tensor<4x8x16x16xf16>
}
