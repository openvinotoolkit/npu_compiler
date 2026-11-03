//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --fold-avgpool-channel-expand %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FoldNoWidthRemainder
module @FoldNoWidthRemainder {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x8x32xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
        input_padding = [0, 13, 0, 0],
        output_padding = [0, 0, 0, 0]
    } : tensor<1x16x8x32xf16, {order = #NHWC}> -> tensor<1x3x8x32xf16, {order = #NHWC}>
    return %pool : tensor<1x3x8x32xf16, {order = #NHWC}>
}

// No leftover channel-alignment Expand: the only Expand left, if any, would be a width-alignment one,
// but here width (32) is already a multiple of the fold factor (16), so no Expand survives at all.
// CHECK-NOT:   VPU.Expand
// CHECK:       [[IN_CAST:%.+]] = VPU.ShapeCast {shape = [1, 48, 8, 2]} inputs([[ARG0]] : tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[IN_CAST]])
// CHECK-NOT:   input_padding
// CHECK-NOT:   output_padding
// CHECK-SAME:  -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[OUT_CAST:%.+]] = VPU.ShapeCast {shape = [1, 3, 8, 32]} inputs([[POOL]] : tensor<1x48x8x2xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK-NOT:   VPU.Slice
// CHECK:       return [[OUT_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FoldWithWidthRemainder
module @FoldWithWidthRemainder {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x8x20xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x3x8x20xf16, {order = #NHWC}>) -> tensor<1x3x8x20xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x20xf16, {order = #NHWC}> -> tensor<1x16x8x20xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
        input_padding = [0, 13, 0, 0],
        output_padding = [0, 0, 0, 0]
    } : tensor<1x16x8x20xf16, {order = #NHWC}> -> tensor<1x3x8x20xf16, {order = #NHWC}>
    return %pool : tensor<1x3x8x20xf16, {order = #NHWC}>
}

// Original 3->16 channel Expand is gone; only the new width-only Expand (20->32) remains.
// CHECK-NOT:   pads_end = [0, 13, 0, 0]
// CHECK:       [[WIDTH_EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]}
// CHECK-SAME:  tensor<1x3x8x20xf16, {order = #NHWC}> -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK:       [[IN_CAST:%.+]] = VPU.ShapeCast {shape = [1, 48, 8, 2]} inputs([[WIDTH_EXPAND]] : tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[IN_CAST]])
// CHECK-NOT:   input_padding
// CHECK-NOT:   output_padding
// CHECK-SAME:  -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[OUT_CAST:%.+]] = VPU.ShapeCast {shape = [1, 3, 8, 32]} inputs([[POOL]] : tensor<1x48x8x2xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK:       [[SLICE:%.+]] = VPU.Slice [[OUT_CAST]] [0, 0, 0, 0] [1, 3, 8, 20] : tensor<1x3x8x32xf16, {order = #NHWC}> to tensor<1x3x8x20xf16, {order = #NHWC}>
// CHECK:       return [[SLICE]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Weight table present: folding would silently corrupt per-channel scale/bias semantics, so the
// pattern must bail out and leave the original Expand + AveragePool untouched.
// CHECK-LABEL: @BailWeightTable
module @BailWeightTable {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x8x32xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}> {
    %scale = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand, %scale) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
        input_padding = [0, 13, 0, 0],
        output_padding = [0, 0, 0, 0]
    } -> tensor<1x3x8x32xf16, {order = #NHWC}>
    return %pool : tensor<1x3x8x32xf16, {order = #NHWC}>
}

// CHECK-NOT:   VPU.ShapeCast
// CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[EXPAND]], %cst)
// CHECK-SAME:  input_padding = [0, 13, 0, 0]
// CHECK-SAME:  output_padding = [0, 0, 0, 0]
// CHECK:       return [[POOL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Channel padding is already cheap (15->16, ~6.7% overhead) while width-folding would need a much
// more expensive 8->16 width pad (100% overhead), so the pattern must bail out on the cost check.
// CHECK-LABEL: @BailCostNotCheaper
module @BailCostNotCheaper {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x15x8x8xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x15x8x8xf16, {order = #NHWC}>) -> tensor<1x15x8x8xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
        : tensor<1x15x8x8xf16, {order = #NHWC}> -> tensor<1x16x8x8xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>,
        input_padding = [0, 1, 0, 0],
        output_padding = [0, 0, 0, 0]
    } : tensor<1x16x8x8xf16, {order = #NHWC}> -> tensor<1x15x8x8xf16, {order = #NHWC}>
    return %pool : tensor<1x15x8x8xf16, {order = #NHWC}>
}

// CHECK-NOT:   VPU.ShapeCast
// CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[EXPAND]])
// CHECK-SAME:  input_padding = [0, 1, 0, 0]
// CHECK-SAME:  output_padding = [0, 0, 0, 0]
// CHECK:       return [[POOL]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// The pool outputs the aligned number of channels, which a Slice cuts off: the Slice is replaced by the folded result
// CHECK-LABEL: @FoldFollowedBySlice
module @FoldFollowedBySlice {

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x8x32xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } : tensor<1x16x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %slice = VPU.Slice %pool [0, 0, 0, 0] [1, 3, 8, 32]
        : tensor<1x16x8x32xf16, {order = #NHWC}> to tensor<1x3x8x32xf16, {order = #NHWC}>
    return %slice : tensor<1x3x8x32xf16, {order = #NHWC}>
}

// CHECK-NOT:   VPU.Expand
// CHECK:       [[IN_CAST:%.+]] = VPU.ShapeCast {shape = [1, 48, 8, 2]} inputs([[ARG0]] : tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[IN_CAST]])
// CHECK-SAME:  -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[OUT_CAST:%.+]] = VPU.ShapeCast {shape = [1, 3, 8, 32]} inputs([[POOL]] : tensor<1x48x8x2xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK-NOT:   VPU.Slice
// CHECK:       return [[OUT_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// The pool outputs the aligned number of channels, which a Slice cuts off, and the width is not a multiple of the fold
// factor: the channel Slice is replaced by the width Slice and the unused pool is erased
// CHECK-LABEL: @FoldFollowedBySliceWithWidthRemainder
module @FoldFollowedBySliceWithWidthRemainder {

// CHECK-LABEL:   @main
// CHECK-SAME:    [[ARG0:%.+]]: tensor<1x3x8x20xf16, {order = #NHWC}>
func.func @main(%arg0: tensor<1x3x8x20xf16, {order = #NHWC}>) -> tensor<1x3x8x20xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x20xf16, {order = #NHWC}> -> tensor<1x16x8x20xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } : tensor<1x16x8x20xf16, {order = #NHWC}> -> tensor<1x16x8x20xf16, {order = #NHWC}>
    %slice = VPU.Slice %pool [0, 0, 0, 0] [1, 3, 8, 20]
        : tensor<1x16x8x20xf16, {order = #NHWC}> to tensor<1x3x8x20xf16, {order = #NHWC}>
    return %slice : tensor<1x3x8x20xf16, {order = #NHWC}>
}

// CHECK-NOT:   pads_end = [0, 13, 0, 0]
// CHECK:       [[WIDTH_EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 12]}
// CHECK-SAME:  tensor<1x3x8x20xf16, {order = #NHWC}> -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK:       [[IN_CAST:%.+]] = VPU.ShapeCast {shape = [1, 48, 8, 2]} inputs([[WIDTH_EXPAND]] : tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[POOL:%.+]] = VPU.NCE.AveragePool([[IN_CAST]])
// CHECK-SAME:  -> tensor<1x48x8x2xf16, {order = #NHWC}>
// CHECK:       [[OUT_CAST:%.+]] = VPU.ShapeCast {shape = [1, 3, 8, 32]} inputs([[POOL]] : tensor<1x48x8x2xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}>
// CHECK:       [[SLICE:%.+]] = VPU.Slice [[OUT_CAST]] [0, 0, 0, 0] [1, 3, 8, 20] : tensor<1x3x8x32xf16, {order = #NHWC}> to tensor<1x3x8x20xf16, {order = #NHWC}>
// CHECK:       return [[SLICE]]
// The original pool with the aligned channels is erased: there is a single pool and a single Slice
// CHECK-NOT:   VPU.NCE.AveragePool
// CHECK-NOT:   VPU.Slice
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A Slice, which does not cut the channels off from the start, does not allow to fold
// CHECK-LABEL: @BailSliceWithOffset
module @BailSliceWithOffset {

// CHECK-LABEL:   @main
func.func @main(%arg0: tensor<1x3x8x32xf16, {order = #NHWC}>) -> tensor<1x3x8x32xf16, {order = #NHWC}> {
    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]}
        : tensor<1x3x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %pool = VPU.NCE.AveragePool(%expand) {
        kernel_size = [1, 1],
        strides = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } : tensor<1x16x8x32xf16, {order = #NHWC}> -> tensor<1x16x8x32xf16, {order = #NHWC}>
    %slice = VPU.Slice %pool [0, 1, 0, 0] [1, 3, 8, 32]
        : tensor<1x16x8x32xf16, {order = #NHWC}> to tensor<1x3x8x32xf16, {order = #NHWC}>
    return %slice : tensor<1x3x8x32xf16, {order = #NHWC}>
}

// CHECK:       VPU.Expand
// CHECK:       VPU.NCE.AveragePool
// CHECK:       VPU.Slice
// CHECK-NOT:   VPU.ShapeCast
}
