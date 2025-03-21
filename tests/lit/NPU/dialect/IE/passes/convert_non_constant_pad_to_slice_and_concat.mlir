//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-non-constant-pad-to-slice-and-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: func.func @convert2DReflectPad([[INPUT:%.+]]: tensor<1x8x13x29xf16>)
func.func @convert2DReflectPad(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>

    return %0 : tensor<1x8x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 1] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 27] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x8x13x1xf16>, tensor<1x8x13x1xf16>, tensor<1x8x13x29xf16>, tensor<1x8x13x1xf16> -> tensor<1x8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 1, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 11, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 10, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x8x1x32xf16>, tensor<1x8x13x32xf16>, tensor<1x8x1x32xf16>, tensor<1x8x1x32xf16> -> tensor<1x8x16x32xf16>

    // CHECK:       return [[CONCAT1]] : tensor<1x8x16x32xf16>
}

// -----

// CHECK: func.func @fuse2DReflectPadwith3DType([[INPUT:%.+]]: tensor<8x13x29xf16>)
func.func @fuse2DReflectPadwith3DType(%arg0: tensor<8x13x29xf16>) -> tensor<8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 1, 2], pads_end_attr = [0, 2, 1]}
                        : tensor<8x13x29xf16> -> tensor<8x16x32xf16>

    return %0 : tensor<8x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 2] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 1] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 27] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<8x13x1xf16>, tensor<8x13x1xf16>, tensor<8x13x29xf16>, tensor<8x13x1xf16> -> tensor<8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 1, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 11, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 10, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 1 : i64>} : tensor<8x1x32xf16>, tensor<8x13x32xf16>, tensor<8x1x32xf16>, tensor<8x1x32xf16> -> tensor<8x16x32xf16>

    // CHECK:       return [[CONCAT1]] : tensor<8x16x32xf16>
}

// -----

// CHECK: func.func @convert4DReflectPad([[INPUT:%.+]]: tensor<16x8x13x29xf16>)
func.func @convert4DReflectPad(%arg0: tensor<16x8x13x29xf16>) -> tensor<19x11x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [2, 1, 1, 2], pads_end_attr = [1, 2, 2, 1]}
                        : tensor<16x8x13x29xf16> -> tensor<19x11x16x32xf16>

    return %0 : tensor<19x11x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 1] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 27] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 3 : i64>} : tensor<16x8x13x1xf16>, tensor<16x8x13x1xf16>, tensor<16x8x13x29xf16>, tensor<16x8x13x1xf16> -> tensor<16x8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 1, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 11, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 10, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<16x8x1x32xf16>, tensor<16x8x13x32xf16>, tensor<16x8x1x32xf16>, tensor<16x8x1x32xf16> -> tensor<16x8x16x32xf16>

    // CHECK:       [[SLICE6:%.+]] = IE.Slice [[CONCAT1]] [0, 1, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[SLICE7:%.+]] = IE.Slice [[CONCAT1]] [0, 6, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[SLICE8:%.+]] = IE.Slice [[CONCAT1]] [0, 5, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[SLICE6]], [[CONCAT1]], [[SLICE7]], [[SLICE8]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x1x16x32xf16>, tensor<16x8x16x32xf16>, tensor<16x1x16x32xf16>, tensor<16x1x16x32xf16> -> tensor<16x11x16x32xf16>

    // CHECK:       [[SLICE9:%.+]] = IE.Slice [[CONCAT2]] [2, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[SLICE10:%.+]] = IE.Slice [[CONCAT2]] [1, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[SLICE11:%.+]] = IE.Slice [[CONCAT2]] [14, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[CONCAT3:%.+]] = IE.Concat([[SLICE9]], [[SLICE10]], [[CONCAT2]], [[SLICE11]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x11x16x32xf16>, tensor<1x11x16x32xf16>, tensor<16x11x16x32xf16>, tensor<1x11x16x32xf16> -> tensor<19x11x16x32xf16>

    // CHECK:       return [[CONCAT3]] : tensor<19x11x16x32xf16>
}

// -----

// CHECK: func.func @convert2DEdgePad([[INPUT:%.+]]: tensor<1x8x13x29xf16>)
func.func @convert2DEdgePad(%arg0: tensor<1x8x13x29xf16>) -> tensor<1x8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]}
                        : tensor<1x8x13x29xf16> -> tensor<1x8x16x32xf16>

    return %0 : tensor<1x8x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 28] [1, 8, 13, 1] : tensor<1x8x13x29xf16> to tensor<1x8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x8x13x1xf16>, tensor<1x8x13x1xf16>, tensor<1x8x13x29xf16>, tensor<1x8x13x1xf16> -> tensor<1x8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 12, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 12, 0] [1, 8, 1, 32] : tensor<1x8x13x32xf16> to tensor<1x8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x8x1x32xf16>, tensor<1x8x13x32xf16>, tensor<1x8x1x32xf16>, tensor<1x8x1x32xf16> -> tensor<1x8x16x32xf16>

    // CHECK:       return [[CONCAT1]] : tensor<1x8x16x32xf16>
}

// -----

// CHECK: func.func @fuse2DEdgePadwith3DType([[INPUT:%.+]]: tensor<8x13x29xf16>)
func.func @fuse2DEdgePadwith3DType(%arg0: tensor<8x13x29xf16>) -> tensor<8x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 1, 2], pads_end_attr = [0, 2, 1]}
                        : tensor<8x13x29xf16> -> tensor<8x16x32xf16>

    return %0 : tensor<8x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 28] [8, 13, 1] : tensor<8x13x29xf16> to tensor<8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<8x13x1xf16>, tensor<8x13x1xf16>, tensor<8x13x29xf16>, tensor<8x13x1xf16> -> tensor<8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 12, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 12, 0] [8, 1, 32] : tensor<8x13x32xf16> to tensor<8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 1 : i64>} : tensor<8x1x32xf16>, tensor<8x13x32xf16>, tensor<8x1x32xf16>, tensor<8x1x32xf16> -> tensor<8x16x32xf16>

    // CHECK:       return [[CONCAT1]] : tensor<8x16x32xf16>
}

// -----

// CHECK: func.func @convert4DEdgePad([[INPUT:%.+]]: tensor<16x8x13x29xf16>)
func.func @convert4DEdgePad(%arg0: tensor<16x8x13x29xf16>) -> tensor<19x11x16x32xf16> {
    %0 = IE.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [2, 1, 1, 2], pads_end_attr = [1, 2, 2, 1]}
                        : tensor<16x8x13x29xf16> -> tensor<19x11x16x32xf16>

    return %0 : tensor<19x11x16x32xf16>

    // CHECK-NOT:      IE.Pad
    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 28] [16, 8, 13, 1] : tensor<16x8x13x29xf16> to tensor<16x8x13x1xf16>
    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[SLICE0]], [[SLICE1]], [[INPUT]], [[SLICE2]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 3 : i64>} : tensor<16x8x13x1xf16>, tensor<16x8x13x1xf16>, tensor<16x8x13x29xf16>, tensor<16x8x13x1xf16> -> tensor<16x8x13x32xf16>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 0, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 12, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[CONCAT0]] [0, 0, 12, 0] [16, 8, 1, 32] : tensor<16x8x13x32xf16> to tensor<16x8x1x32xf16>
    // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[SLICE3]], [[CONCAT0]], [[SLICE4]], [[SLICE5]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 2 : i64>} : tensor<16x8x1x32xf16>, tensor<16x8x13x32xf16>, tensor<16x8x1x32xf16>, tensor<16x8x1x32xf16> -> tensor<16x8x16x32xf16>

    // CHECK:       [[SLICE6:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[SLICE7:%.+]] = IE.Slice [[CONCAT1]] [0, 7, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[SLICE8:%.+]] = IE.Slice [[CONCAT1]] [0, 7, 0, 0] [16, 1, 16, 32] : tensor<16x8x16x32xf16> to tensor<16x1x16x32xf16>
    // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[SLICE6]], [[CONCAT1]], [[SLICE7]], [[SLICE8]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x1x16x32xf16>, tensor<16x8x16x32xf16>, tensor<16x1x16x32xf16>, tensor<16x1x16x32xf16> -> tensor<16x11x16x32xf16>

    // CHECK:       [[SLICE9:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[SLICE10:%.+]] = IE.Slice [[CONCAT2]] [0, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[SLICE11:%.+]] = IE.Slice [[CONCAT2]] [15, 0, 0, 0] [1, 11, 16, 32] : tensor<16x11x16x32xf16> to tensor<1x11x16x32xf16>
    // CHECK:       [[CONCAT3:%.+]] = IE.Concat([[SLICE9]], [[SLICE10]], [[CONCAT2]], [[SLICE11]]) {
    // CHECK-SAME:          per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x11x16x32xf16>, tensor<1x11x16x32xf16>, tensor<16x11x16x32xf16>, tensor<1x11x16x32xf16> -> tensor<19x11x16x32xf16>

    // CHECK:       return [[CONCAT3]] : tensor<19x11x16x32xf16>
}
