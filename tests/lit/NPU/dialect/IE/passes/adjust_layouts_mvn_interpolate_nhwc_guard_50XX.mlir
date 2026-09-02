//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-se-ptrs-operations=true" --adjust-layouts --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// MVNLayoutInfoOpModelForSW's NPU50XX-only guard: when MVN's parent is a NEAREST-mode
// NCE-compatible IE.Interpolate (real pattern from a production upsample+conditioning-norm
// block: channel count aligned to VPU_CHANNEL_ALIGNMENT*numTiles, tensor large enough
// relative to per-tile CMX -- see VPU::NCEInvariant::isLargeEnoughForDPUOverSHAVE), NHWC is
// kept so the MVN feeds RunMVNNormalizeOnDPU's DPU path directly, instead of round-tripping
// through NCHW only for RunMVNNormalizeOnDPU to require NHWC again immediately after.
// Verified against the real compiler: with these channel/CMX values, no Reorder is inserted
// around IE.MVN at all.
// CHECK-LABEL: @KeepNHWCForNCEInterpolateToMVN
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x192x16x48xf16, {order = #NHWC}>)
func.func @KeepNHWCForNCEInterpolateToMVN(%arg0: tensor<1x192x16x48xf16, {order = #NHWC}>) -> tensor<1x192x32x96xf16, {order = #NHWC}> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 96]
         } : tensor<1x192x16x48xf16, {order = #NHWC}> -> tensor<1x192x32x96xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
         : tensor<1x192x32x96xf16, {order = #NHWC}> -> tensor<1x192x32x96xf16, {order = #NHWC}>
    return %1 : tensor<1x192x32x96xf16, {order = #NHWC}>

    // CHECK:        [[INTERP:%.+]] = IE.Interpolate([[ARG0]])
    // CHECK-SAME:       -> tensor<1x192x32x96xf16, {order = #NHWC}>

    // CHECK-NOT:    IE.Reorder

    // CHECK:        [[MVN:%.+]] = IE.MVN([[INTERP]])
    // CHECK-SAME:       : tensor<1x192x32x96xf16, {order = #NHWC}> -> tensor<1x192x32x96xf16, {order = #NHWC}>

    // CHECK-NOT:    IE.Reorder

    // CHECK:        return [[MVN]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Negative case: same NCE-compatible NEAREST Interpolate parent, but channel count (16) is
// below VPU_CHANNEL_ALIGNMENT(16) * numTiles(3) = 48, so the guard's first condition fails
// and the pre-existing isMVNEfficientWithNCHWLayout heuristic switches to NCHW as before.
// Verified against the real compiler: Reorder-to-NCHW is inserted before MVN and
// Reorder-back-to-NHWC after, to match the function's declared NHWC boundary type.
// CHECK-LABEL: @SmallChannelsNoGuard
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x16x16x48xf16, {order = #NHWC}>)
func.func @SmallChannelsNoGuard(%arg0: tensor<1x16x16x48xf16, {order = #NHWC}>) -> tensor<1x16x32x96xf16, {order = #NHWC}> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [32, 96]
         } : tensor<1x16x16x48xf16, {order = #NHWC}> -> tensor<1x16x32x96xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true}
         : tensor<1x16x32x96xf16, {order = #NHWC}> -> tensor<1x16x32x96xf16, {order = #NHWC}>
    return %1 : tensor<1x16x32x96xf16, {order = #NHWC}>

    // CHECK:        [[INTERP:%.+]] = IE.Interpolate([[ARG0]])
    // CHECK-SAME:       -> tensor<1x16x32x96xf16, {order = #NHWC}>

    // CHECK:        [[REORDER_IN:%.+]] = IE.Reorder([[INTERP]]) {dstOrder = #NCHW}
    // CHECK-SAME:       -> tensor<1x16x32x96xf16>

    // CHECK:        [[MVN:%.+]] = IE.MVN([[REORDER_IN]])
    // CHECK-SAME:       : tensor<1x16x32x96xf16> -> tensor<1x16x32x96xf16>

    // CHECK:        [[REORDER_OUT:%.+]] = IE.Reorder([[MVN]]) {dstOrder = #NHWC}
    // CHECK-SAME:       -> tensor<1x16x32x96xf16, {order = #NHWC}>

    // CHECK:        return [[REORDER_OUT]]
}
