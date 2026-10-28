//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-se-ptrs-operations=true" --lower-ops-to-se-nce %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotLowerPadOpWith32ICToNCE
// CHECK-SAME: ([[INPUT_DATA:%.+]]: tensor<1x32x262x262xf16, {order = #NHWC}>) -> tensor<1x32x264x264xf16, {order = #NHWC}> {
func.func @DoNotLowerPadOpWith32ICToNCE(%input: tensor<1x32x262x262xf16, {order = #NHWC}>) -> tensor<1x32x264x264xf16, {order = #NHWC}> {
    %0 = VPU.Pad(%input) {
            mode = #IE.pad_mode<REFLECT>,
            pad_value_attr = 0.000000e+00 : f64,
            pads_begin_attr = [0, 0, 1, 1],
            pads_end_attr = [0, 0, 1, 1]
        } : tensor<1x32x262x262xf16, {order = #NHWC}> -> tensor<1x32x264x264xf16, {order = #NHWC}>

    return %0 : tensor<1x32x264x264xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Pad([[INPUT_DATA]]) {
    // CHECK-SAME:      mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 1, 1]} :
    // CHECK-SAME:          tensor<1x32x262x262xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x32x264x264xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x264x264xf16, {order = #NHWC}>
}
