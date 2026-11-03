//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Case B (after PermuteCast propagation): shape is expressed in canonical NCHW
// index order (Y = [N, 1, H, W], UV = [N, 2, H/2, W/2]) while order = NHWC.
// Here the semantic channel is the regular C dimension, so no re-mapping of
// cDim/hDim is required for the already-split multi-input case: the output
// shape/order must be preserved as-is.
// CHECK-LABEL: @AlreadySplitInputsWithNHWCOrder
// CHECK-SAME: ([[Y:%[^:]+]]: tensor<1x1x180x1280xf16, {order = #NHWC}>, [[UV:%[^:]+]]: tensor<1x2x90x640xf16, {order = #NHWC}>)
func.func @AlreadySplitInputsWithNHWCOrder(
        %arg0: tensor<1x1x180x1280xf16, {order = #NHWC}>,
        %arg1: tensor<1x2x90x640xf16, {order = #NHWC}>) -> tensor<1x3x180x1280xf16, {order = #NHWC}> {
    %0 = VPU.YuvToRgb(%arg0, %arg1) {
        inFmt = #IE.color_fmt<NV12>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        operandSegmentSizes = array<i32: 1, 1, 0>,
        outFmt = #IE.color_fmt<BGR>,
        scale = 0.00373366661 : f32
    } : tensor<1x1x180x1280xf16, {order = #NHWC}>, tensor<1x2x90x640xf16, {order = #NHWC}>
        -> tensor<1x3x180x1280xf16, {order = #NHWC}>
    return %0 : tensor<1x3x180x1280xf16, {order = #NHWC}>

    // CHECK: [[YUV_TO_RGB:%.+]] = VPU.YuvToRgb([[Y]], [[UV]])
    // CHECK-SAME{LITERAL}: operandSegmentSizes = array<i32: 1, 1, 0>
    // CHECK-SAME: : tensor<1x1x180x1280xf16, {order = #NHWC}>, tensor<1x2x90x640xf16, {order = #NHWC}>
    // CHECK-SAME: -> tensor<1x3x180x1280xf16, {order = #NHWC}>
    // CHECK: return [[YUV_TO_RGB]] : tensor<1x3x180x1280xf16, {order = #NHWC}>
}
