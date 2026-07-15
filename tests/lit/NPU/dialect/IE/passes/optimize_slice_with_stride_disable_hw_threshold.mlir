//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" \
// RUN:          --optimize-slice-with-stride="disable-min-hw-threshold=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertSliceSmallTensorWhenThresholdDisabled
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x32x32xf32, {order = #NHWC}>)
func.func @ConvertSliceSmallTensorWhenThresholdDisabled(%arg0: tensor<1x2x32x32xf32, {order = #NHWC}>)
    -> tensor<1x1x32x32xf16, {order = #NHWC}> {
    %CONVERT = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x2x32x32xf32, {order = #NHWC}> -> tensor<1x2x32x32xf16, {order = #NHWC}>
    %SLICE = IE.Slice %CONVERT [0, 1, 0, 0] [1, 1, 32, 32]
        : tensor<1x2x32x32xf16, {order = #NHWC}> to tensor<1x1x32x32xf16, {order = #NHWC}>
    return %SLICE : tensor<1x1x32x32xf16, {order = #NHWC}>

    // CHECK-NOT:   IE.Slice
    // CHECK:       IE.Convolution
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ConvertRank3SliceSmallTensorWhenThresholdDisabled
// CHECK-SAME: ([[INPUT:%.+]]: tensor<2x160x3xf16>)
func.func @ConvertRank3SliceSmallTensorWhenThresholdDisabled(%arg0: tensor<2x160x3xf16>)
    -> tensor<2x160x1xf16> {
    %0 = IE.Slice %arg0 [0, 0, 2] [2, 160, 1] : tensor<2x160x3xf16> to tensor<2x160x1xf16>
    return %0 : tensor<2x160x1xf16>

    // CHECK-NOT:   IE.Slice
    // CHECK:       IE.Convolution
}
