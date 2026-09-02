//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-vpunn-cost-for-tiling=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010



// -----

// E220898: In HostCompile mode an NCE operation that carries a Weight Table must not be tiled over
// channels. Channel tiling makes SCF tiling slice the Weight Table per tile and pass it into the
// outlined compute kernel as a function argument, which the PatchWeightsTable pass cannot patch.
// This DepthConvolution carries a Weight Table and would tile over channels ([1, 2, 1, 1]) without
// the workaround; the workaround forces spatial (height) tiling so the Weight Table stays whole,
// keeping channels untiled.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @DepthConvWeightsTableKeepWholeInHostCompile(%arg0: tensor<1x96x112x112xf16, {order = #NHWC}>) -> tensor<1x96x56x56xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weight_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %weight_table) rawFilterShape [96, 1, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [2, 2]
    } -> tensor<1x96x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x96x56x56xf16, {order = #NHWC}>

    // The Weight Table is kept whole: the strategy is 4D, dim 0 (N) and dim 1 (C) are both 1
    // (channels NOT tiled), and at least one spatial dim (H or W) is > 1 so the op IS tiled.
    // CHECK:      VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 1, {{([2-9]|[1-9][0-9]+), [0-9]+|[0-9]+, ([2-9]|[1-9][0-9]+)}}]
}
