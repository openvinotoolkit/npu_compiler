//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --expand-activation-channels="se-experimental-ops-enabled=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ExpandPadOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x20x23x30xf16, {order = #NHWC}>)
func.func @ExpandPadOp(%input: tensor<1x20x23x30xf16, {order = #NHWC}>) -> tensor<1x20x26x33xf16, {order = #NHWC}> {
    %0 = IE.Pad(%input) {
                mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64,
                pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
            } : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x20x26x33xf16, {order = #NHWC}>

    return %0 : tensor<1x20x26x33xf16, {order = #NHWC}>

    // CHECK:       [[EXPAND:%.+]] = IE.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]}
    // CHECK-SAME:      : tensor<1x20x23x30xf16, {order = #NHWC}> -> tensor<1x32x23x30xf16, {order = #NHWC}>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[EXPAND]]) {
    // CHECK-SAME:          mode = #IE.pad_mode<REFLECT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 2], pads_end_attr = [0, 0, 2, 1]
    // CHECK-SAME:      } : tensor<1x32x23x30xf16, {order = #NHWC}> -> tensor<1x32x26x33xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT_SLICE:%.+]] = IE.Slice [[PAD]] [0, 0, 0, 0] [1, 20, 26, 33]
    // CHECK-SAME:      : tensor<1x32x26x33xf16, {order = #NHWC}> to tensor<1x20x26x33xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT_SLICE]]
}
