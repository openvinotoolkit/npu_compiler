//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertToMultiInputsNV12
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x552x432xf16, {order = #NHWC}>)
func.func @ConvertToMultiInputsNV12(%arg0: tensor<1x1x552x432xf16, {order = #NHWC}>) -> tensor<1x3x368x432xf16, {order = #NHWC}> {
    %0 = IE.YuvToRgb(%arg0) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 0, 0>, outFmt = #IE.color_fmt<RGB>}
        : tensor<1x1x552x432xf16, {order = #NHWC}> -> tensor<1x3x368x432xf16, {order = #NHWC}>
    return %0 : tensor<1x3x368x432xf16, {order = #NHWC}>

    // CHECK: [[LUMA:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 368, 432] : tensor<1x1x552x432xf16, {order = #NHWC}> to tensor<1x1x368x432xf16, {order = #NHWC}>
    // CHECK: [[CHROMA:%.+]] = IE.Slice [[ARG_0]] [0, 0, 368, 0] [1, 1, 184, 432] : tensor<1x1x552x432xf16, {order = #NHWC}> to tensor<1x1x184x432xf16, {order = #NHWC}>
    // CHECK: [[RESHAPE:%.+]] = IE.Reshape([[CHROMA]]) {shape_value = [1, 2, 184, 216]} : tensor<1x1x184x432xf16, {order = #NHWC}> -> tensor<1x2x184x216xf16>
    // CHECK: [[CAST:%.+]] = IE.LayoutCast([[RESHAPE]]) {dst_order = #NHWC} : tensor<1x2x184x216xf16> -> tensor<1x2x184x216xf16, {order = #NHWC}>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[LUMA]], [[CAST]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x1x368x432xf16, {order = #NHWC}>, tensor<1x2x184x216xf16, {order = #NHWC}> -> tensor<1x3x368x432xf16, {order = #NHWC}>
    // CHECK: return [[YUV_TO_RGB]] : tensor<1x3x368x432xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertToMultiInputsI420
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1x552x432xf16, {order = #NHWC}>)
func.func @ConvertToMultiInputsI420(%arg0: tensor<1x1x552x432xf16, {order = #NHWC}>) -> tensor<1x3x368x432xf16, {order = #NHWC}> {
    %0 = IE.YuvToRgb(%arg0) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 0, 0>, outFmt = #IE.color_fmt<RGB>}
        : tensor<1x1x552x432xf16, {order = #NHWC}> -> tensor<1x3x368x432xf16, {order = #NHWC}>
    return %0 : tensor<1x3x368x432xf16, {order = #NHWC}>

    // CHECK: [[LUMA:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 1, 368, 432] : tensor<1x1x552x432xf16, {order = #NHWC}> to tensor<1x1x368x432xf16, {order = #NHWC}>
    // CHECK: [[UV:%.+]] = IE.Slice [[ARG_0]] [0, 0, 368, 0] [1, 1, 184, 432] : tensor<1x1x552x432xf16, {order = #NHWC}> to tensor<1x1x184x432xf16, {order = #NHWC}>
    // CHECK: [[UV_RESHAPE:%.+]] = IE.Reshape([[UV]]) {shape_value = [1, 1, 368, 216]} : tensor<1x1x184x432xf16, {order = #NHWC}> -> tensor<1x1x368x216xf16>
    // CHECK: [[UV_CAST:%.+]] = IE.LayoutCast([[UV_RESHAPE]]) {dst_order = #NHWC} : tensor<1x1x368x216xf16> -> tensor<1x1x368x216xf16, {order = #NHWC}>
    // CHECK: [[U:%.+]] = IE.Slice [[UV_CAST]] [0, 0, 0, 0] [1, 1, 184, 216] : tensor<1x1x368x216xf16, {order = #NHWC}> to tensor<1x1x184x216xf16, {order = #NHWC}>
    // CHECK: [[V:%.+]] = IE.Slice [[UV_CAST]] [0, 0, 184, 0] [1, 1, 184, 216] : tensor<1x1x368x216xf16, {order = #NHWC}> to tensor<1x1x184x216xf16, {order = #NHWC}>
    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[LUMA]], [[U]], [[V]]) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 1, 1>, outFmt = #IE.color_fmt<RGB>} : tensor<1x1x368x432xf16, {order = #NHWC}>, tensor<1x1x184x216xf16, {order = #NHWC}>, tensor<1x1x184x216xf16, {order = #NHWC}> -> tensor<1x3x368x432xf16, {order = #NHWC}>
    // CHECK: return [[YUV_TO_RGB]] : tensor<1x3x368x432xf16, {order = #NHWC}>
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AlreadySplitInputsWithNHWCOrder
// CHECK-SAME: ([[Y:%[^:]+]]: tensor<1x1x180x1280xf16, {order = #NHWC}>, [[UV:%[^:]+]]: tensor<1x2x90x640xf16, {order = #NHWC}>)
func.func @AlreadySplitInputsWithNHWCOrder(
        %arg0: tensor<1x1x180x1280xf16, {order = #NHWC}>,
        %arg1: tensor<1x2x90x640xf16, {order = #NHWC}>) -> tensor<1x3x180x1280xf16, {order = #NHWC}> {
    %0 = IE.YuvToRgb(%arg0, %arg1) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>}
        : tensor<1x1x180x1280xf16, {order = #NHWC}>, tensor<1x2x90x640xf16, {order = #NHWC}> -> tensor<1x3x180x1280xf16, {order = #NHWC}>
    return %0 : tensor<1x3x180x1280xf16, {order = #NHWC}>

    // CHECK: [[YUV_TO_RGB:%.+]] = IE.YuvToRgb([[Y]], [[UV]])
    // CHECK-SAME{LITERAL}: operandSegmentSizes = array<i32: 1, 1, 0>
    // CHECK-SAME: : tensor<1x1x180x1280xf16, {order = #NHWC}>, tensor<1x2x90x640xf16, {order = #NHWC}> -> tensor<1x3x180x1280xf16, {order = #NHWC}>
    // CHECK: return [[YUV_TO_RGB]] : tensor<1x3x180x1280xf16, {order = #NHWC}>
}
