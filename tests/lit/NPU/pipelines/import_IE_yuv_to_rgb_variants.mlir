//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-translate --platform=%platform% --import-IE ./IR/yuv_to_rgb_import.xml -o %t
// RUN: FileCheck %s --input-file %t
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Verifying that imported YUV->RGB/BGR conversion is wrapped as NWCH input transpose, YuvToRgb, and NHWC output transpose.

// CHECK: [[I420B_Y_T:%.+]] = IE.Transpose([[I420B_Y:%.+]]) {order_value = #NWCH} : tensor<1x4x8x1xui8> -> tensor<1x1x4x8xui8>
// CHECK: [[I420B_U_T:%.+]] = IE.Transpose([[I420B_U:%.+]]) {order_value = #NWCH} : tensor<1x2x4x1xui8> -> tensor<1x1x2x4xui8>
// CHECK: [[I420B_V_T:%.+]] = IE.Transpose([[I420B_V:%.+]]) {order_value = #NWCH} : tensor<1x2x4x1xui8> -> tensor<1x1x2x4xui8>
// CHECK: [[I420B_BGR:%.+]] = IE.YuvToRgb([[I420B_Y_T]], [[I420B_U_T]], [[I420B_V_T]]) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 1, 1>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32} : tensor<1x1x4x8xui8>, tensor<1x1x2x4xui8>, tensor<1x1x2x4xui8> -> tensor<1x3x4x8xui8>
// CHECK: [[I420B_BGR_OUT:%.+]] = IE.Transpose([[I420B_BGR]]) {order_value = #NHWC} : tensor<1x3x4x8xui8> -> tensor<1x4x8x3xui8>

// CHECK: [[I420_Y_T:%.+]] = IE.Transpose([[I420_Y:%.+]]) {order_value = #NWCH} : tensor<1x4x8x1xui8> -> tensor<1x1x4x8xui8>
// CHECK: [[I420_U_T:%.+]] = IE.Transpose([[I420_U:%.+]]) {order_value = #NWCH} : tensor<1x2x4x1xui8> -> tensor<1x1x2x4xui8>
// CHECK: [[I420_V_T:%.+]] = IE.Transpose([[I420_V:%.+]]) {order_value = #NWCH} : tensor<1x2x4x1xui8> -> tensor<1x1x2x4xui8>
// CHECK: [[I420_RGB:%.+]] = IE.YuvToRgb([[I420_Y_T]], [[I420_U_T]], [[I420_V_T]]) {inFmt = #IE.color_fmt<I420>, operandSegmentSizes = array<i32: 1, 1, 1>, outFmt = #IE.color_fmt<RGB>, scale = 1.000000e+00 : f32} : tensor<1x1x4x8xui8>, tensor<1x1x2x4xui8>, tensor<1x1x2x4xui8> -> tensor<1x3x4x8xui8>
// CHECK: [[I420_RGB_OUT:%.+]] = IE.Transpose([[I420_RGB]]) {order_value = #NHWC} : tensor<1x3x4x8xui8> -> tensor<1x4x8x3xui8>

// CHECK: [[NV12B_Y_T:%.+]] = IE.Transpose([[NV12B_Y:%.+]]) {order_value = #NWCH} : tensor<1x4x8x1xui8> -> tensor<1x1x4x8xui8>
// CHECK: [[NV12B_UV_T:%.+]] = IE.Transpose([[NV12B_UV:%.+]]) {order_value = #NWCH} : tensor<1x2x4x2xui8> -> tensor<1x2x2x4xui8>
// CHECK: [[NV12B_BGR:%.+]] = IE.YuvToRgb([[NV12B_Y_T]], [[NV12B_UV_T]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<BGR>, scale = 1.000000e+00 : f32} : tensor<1x1x4x8xui8>, tensor<1x2x2x4xui8> -> tensor<1x3x4x8xui8>
// CHECK: [[NV12B_BGR_OUT:%.+]] = IE.Transpose([[NV12B_BGR]]) {order_value = #NHWC} : tensor<1x3x4x8xui8> -> tensor<1x4x8x3xui8>

// CHECK: [[NV12_Y_T:%.+]] = IE.Transpose([[NV12_Y:%.+]]) {order_value = #NWCH} : tensor<1x4x8x1xui8> -> tensor<1x1x4x8xui8>
// CHECK: [[NV12_UV_T:%.+]] = IE.Transpose([[NV12_UV:%.+]]) {order_value = #NWCH} : tensor<1x2x4x2xui8> -> tensor<1x2x2x4xui8>
// CHECK: [[NV12_RGB:%.+]] = IE.YuvToRgb([[NV12_Y_T]], [[NV12_UV_T]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, scale = 1.000000e+00 : f32} : tensor<1x1x4x8xui8>, tensor<1x2x2x4xui8> -> tensor<1x3x4x8xui8>
// CHECK: [[NV12_RGB_OUT:%.+]] = IE.Transpose([[NV12_RGB]]) {order_value = #NHWC} : tensor<1x3x4x8xui8> -> tensor<1x4x8x3xui8>

// CHECK: return
