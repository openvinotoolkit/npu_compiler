//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --align-dimensions-for-dpu %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>

// CHECK-LABEL: @ExpandPermuteQuantizeWidth
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x299x299xf16>)
func.func @ExpandPermuteQuantizeWidth(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299x!qElemType, {order = #NHWC}> {
    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x299x299xf16> -> tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    return %PERMUTE_QUANTIZE : tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[ARG_0]]) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 5]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16> -> tensor<1x3x299x304xf16>

    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[EXPAND]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x299x304xf16> -> tensor<1x3x299x304x!qElemType, {order = #NHWC}>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[PERMUTE_QUANTIZE]] [0, 0, 0, 0] [1, 3, 299, 299] :
    // CHECK-SAME:  tensor<1x3x299x304x!qElemType, {order = #NHWC}> to tensor<1x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[SLICE]] : tensor<1x3x299x299x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>

// CHECK-LABEL: @SkipPermuteQuantizeWithBatch2
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<2x3x299x299xf16>)
func.func @SkipPermuteQuantizeWithBatch2(%arg0: tensor<2x3x299x299xf16>) -> tensor<2x3x299x299x!qElemType, {order = #NHWC}> {
    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<2x3x299x299xf16> -> tensor<2x3x299x299x!qElemType, {order = #NHWC}>

    return %PERMUTE_QUANTIZE : tensor<2x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<2x3x299x299xf16> -> tensor<2x3x299x299x!qElemType, {order = #NHWC}>

    // CHECK:   return [[PERMUTE_QUANTIZE]] : tensor<2x3x299x299x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>

// CHECK-LABEL: @SkipPermuteQuantizeToNCWH
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x299x299xf16>)
func.func @SkipPermuteQuantizeToNCWH(%arg0: tensor<1x3x299x299xf16>) -> tensor<1x3x299x299x!qElemType, {order = #NCWH}> {
    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NCWH,
        mem_perm = #NCWH,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x299x299xf16> -> tensor<1x3x299x299x!qElemType, {order = #NCWH}>

    return %PERMUTE_QUANTIZE : tensor<1x3x299x299x!qElemType, {order = #NCWH}>

    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NCWH,
    // CHECK-SAME:      mem_perm = #NCWH,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x299x299xf16> -> tensor<1x3x299x299x!qElemType, {order = #NCWH}>

    // CHECK:   return [[PERMUTE_QUANTIZE]] : tensor<1x3x299x299x!qElemType, {order = #NCWH}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>

// CHECK-LABEL: @SkipWidth1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x3x299x1xf16>)
func.func @SkipWidth1(%arg0: tensor<1x3x299x1xf16>) -> tensor<1x3x299x1x!qElemType, {order = #NHWC}> {
    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x3x299x1xf16> -> tensor<1x3x299x1x!qElemType, {order = #NHWC}>

    return %PERMUTE_QUANTIZE : tensor<1x3x299x1x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x299x1xf16> -> tensor<1x3x299x1x!qElemType, {order = #NHWC}>

    // CHECK:   return [[PERMUTE_QUANTIZE]] : tensor<1x3x299x1x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078407292272530352:128>

// CHECK-LABEL: @SkipWidthFP32
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x2x2xf32>)
func.func @SkipWidthFP32(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x2x2x!qElemType, {order = #NHWC}> {
    %PERMUTE_QUANTIZE = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 0, 0, 0]
    } : tensor<1x1x2x2xf32> -> tensor<1x1x2x2x!qElemType, {order = #NHWC}>

    return %PERMUTE_QUANTIZE : tensor<1x1x2x2x!qElemType, {order = #NHWC}>

    // CHECK:   [[PERMUTE_QUANTIZE:%.+]] = IE.PermuteQuantize([[ARG_0]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 0, 0, 0]
    // CHECK-SAME:  } : tensor<1x1x2x2xf32> -> tensor<1x1x2x2x!qElemType, {order = #NHWC}>

    // CHECK:   return [[PERMUTE_QUANTIZE]] : tensor<1x1x2x2x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ExpandLogSoftmaxWidth
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x151x7049xf16>)
func.func @ExpandLogSoftmaxWidth(%arg0: tensor<1x1x151x7049xf16>) -> tensor<1x1x151x7049xf16> {
    %LOG_SOFTMAX = IE.LogSoftmax(%arg0) {
        axisInd = 3 : i64
    } : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf16>

    return %LOG_SOFTMAX : tensor<1x1x151x7049xf16>

    // CHECK:   [[EXPAND:%.+]] = IE.Expand([[ARG_0]]) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 7]
    // CHECK-SAME:  } : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7056xf16>

    // CHECK:   [[LOG_SOFTMAX:%.+]] = IE.LogSoftmax([[EXPAND]]) {
    // CHECK-SAME:      axisInd = 3 : i64,
    // CHECK-SAME:      padSize = 7 : i64
    // CHECK-SAME:  } : tensor<1x1x151x7056xf16> -> tensor<1x1x151x7056xf16>

    // CHECK:   [[SLICE:%.+]] = IE.Slice [[LOG_SOFTMAX]] [0, 0, 0, 0] [1, 1, 151, 7049] :
    // CHECK-SAME:  tensor<1x1x151x7056xf16> to tensor<1x1x151x7049xf16>

    // CHECK:   return [[SLICE]] : tensor<1x1x151x7049xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipLogSoftmaxNonInnermostAxis
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x151x7049xf16>)
func.func @SkipLogSoftmaxNonInnermostAxis(%arg0: tensor<1x1x151x7049xf16>) -> tensor<1x1x151x7049xf16> {
    %LOG_SOFTMAX = IE.LogSoftmax(%arg0) {
        axisInd = 2 : i64
    } : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf16>

    return %LOG_SOFTMAX : tensor<1x1x151x7049xf16>

    // CHECK:   [[LOG_SOFTMAX:%.+]] = IE.LogSoftmax([[ARG_0]]) {
    // CHECK-SAME:      axisInd = 2 : i64
    // CHECK-SAME:  } : tensor<1x1x151x7049xf16> -> tensor<1x1x151x7049xf16>

    // CHECK:   return [[LOG_SOFTMAX]] : tensor<1x1x151x7049xf16>
}
