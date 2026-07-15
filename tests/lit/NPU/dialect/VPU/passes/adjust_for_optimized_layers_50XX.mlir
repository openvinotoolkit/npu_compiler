//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --adjust-for-optimized-layers %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: AutopadODUAvgPool
module @AutopadODUAvgPool {

  config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
  }

// CHECK-LABEL:   @NotAdjustForAvgPoolWithPermute
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x16x513x513xf16, {order = #NHWC}>
func.func @NotAdjustForAvgPoolWithPermute(%arg0: tensor<1x16x513x513xf16, {order = #NHWC}>) -> tensor<1x3x513x513xf16> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
        quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        input_padding = [0, 13, 0, 0],
        output_padding = [0, 0, 0, 0]
    } -> tensor<1x3x513x513xf16>

    return %0 : tensor<1x3x513x513xf16>

    // CHECK:           [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {
    // CHECK-SAME:              input_padding = [0, 13, 0, 0], kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0],
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:              ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64,
    // CHECK-SAME:              quant_scale = [1.000000e+00], fp_prelu_alpha = 0.199951171875 : f64>,
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x3x513x513xf16>

    // CHECK:           return [[AVGPOOL]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNCHW
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x2x16x32xf16>)
func.func @AdjustForSoftmaxMultiShaveOptNCHW(%arg0: tensor<1x2x16x32xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x2x16x32xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 8, 4, 32]} inputs([[ARG_0]] : tensor<1x2x16x32xf16>) -> tensor<1x8x4x32xf16>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x8x4x32xf16> -> tensor<1x8x4x32xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[SOFTMAX]] : tensor<1x8x4x32xf16>) -> tensor<1x2x16x32xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x2x16xf16, {order = #NHWC}>)
func.func @AdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x32x2x16xf16, {order = #NHWC}> -> tensor<1x32x2x16xf16, {order = #NHWC}>
    return %0 : tensor<1x32x2x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 32, 8, 4]} inputs([[ARG_0]] : tensor<1x32x2x16xf16, {order = #NHWC}>) -> tensor<1x32x8x4xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 1 : i64} : tensor<1x32x8x4xf16, {order = #NHWC}> -> tensor<1x32x8x4xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 32, 2, 16]} inputs([[SOFTMAX]] : tensor<1x32x8x4xf16, {order = #NHWC}>) -> tensor<1x32x2x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotAdjustForSoftmaxMultiShaveOptNHWC
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x3x3xf16, {order = #NHWC}>)
func.func @NotAdjustForSoftmaxMultiShaveOptNHWC(%arg0: tensor<1x64x3x3xf16, {order = #NHWC}>) -> tensor<1x64x3x3xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x3x3xf16, {order = #NHWC}>
    return %0 : tensor<1x64x3x3xf16, {order = #NHWC}>

    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[ARG_0]]) {axisInd = 1 : i64} : tensor<1x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.ShapeCast
    // CHECK:        return [[SOFTMAX]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNCHWwithBatch
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x8x16x16xf16>)
func.func @AdjustForSoftmaxMultiShaveOptNCHWwithBatch(%arg0: tensor<2x8x16x16xf16>) -> tensor<2x8x16x16xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x8x16x16xf16> -> tensor<2x8x16x16xf16>
    return %0 : tensor<2x8x16x16xf16>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 16, 16, 16]} inputs([[ARG_0]] : tensor<2x8x16x16xf16>) -> tensor<1x16x16x16xf16>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x16x16x16xf16> -> tensor<1x16x16x16xf16>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [2, 8, 16, 16]} inputs([[SOFTMAX]] : tensor<1x16x16x16xf16>) -> tensor<2x8x16x16xf16>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AdjustForSoftmaxMultiShaveOptNHWCwithBatch
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<2x8x16x16xf16, {order = #NHWC}>)
func.func @AdjustForSoftmaxMultiShaveOptNHWCwithBatch(%arg0: tensor<2x8x16x16xf16, {order = #NHWC}>) -> tensor<2x8x16x16xf16, {order = #NHWC}> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<2x8x16x16xf16, {order = #NHWC}> -> tensor<2x8x16x16xf16, {order = #NHWC}>
    return %0 : tensor<2x8x16x16xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 8, 32, 16]} inputs([[ARG_0]] : tensor<2x8x16x16xf16, {order = #NHWC}>) -> tensor<1x8x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]]) {axisInd = 3 : i64} : tensor<1x8x32x16xf16, {order = #NHWC}> -> tensor<1x8x32x16xf16, {order = #NHWC}>
    // CHECK:        [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [2, 8, 16, 16]} inputs([[SOFTMAX]] : tensor<1x8x32x16xf16, {order = #NHWC}>) -> tensor<2x8x16x16xf16, {order = #NHWC}>
    // CHECK:        return [[SHAPECAST_OUT]]
}

// -----

// Verify that AdjustShapeForSoftmax also reshapes the precomputed max operand
// consistently with the input shape adjustment.
// input [1,2,16,32] axis=3 → [1,8,4,32] axis=3 (multiShaveOpt for 2 shaves)
// max   [1,2,16, 1]       → [1,8,4, 1]  (axis dim stays 1, other dims follow input)
// CHECK-LABEL: func.func @AdjustSoftMaxWithMaxMultiShave
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x2x16x32xf16>, [[MAX:%.+]]: tensor<1x2x16x1xf16>)
func.func @AdjustSoftMaxWithMaxMultiShave(%arg0: tensor<1x2x16x32xf16>, %max: tensor<1x2x16x1xf16>) -> tensor<1x2x16x32xf16> {
    %0 = VPU.SoftMax(%arg0, %max) {axisInd = 3 : i64} : tensor<1x2x16x32xf16>, tensor<1x2x16x1xf16> -> tensor<1x2x16x32xf16>
    return %0 : tensor<1x2x16x32xf16>

    // CHECK: [[SHAPECAST_IN:%.+]] = VPU.ShapeCast {shape = [1, 8, 4, 32]} inputs([[INPUT]] : tensor<1x2x16x32xf16>) -> tensor<1x8x4x32xf16>
    // CHECK: [[SHAPECAST_MAX:%.+]] = VPU.ShapeCast {shape = [1, 8, 4, 1]} inputs([[MAX]] : tensor<1x2x16x1xf16>) -> tensor<1x8x4x1xf16>
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[SHAPECAST_IN]], [[SHAPECAST_MAX]]) {axisInd = 3 : i64} : tensor<1x8x4x32xf16>, tensor<1x8x4x1xf16> -> tensor<1x8x4x32xf16>
    // CHECK: [[SHAPECAST_OUT:%.+]] = VPU.ShapeCast {shape = [1, 2, 16, 32]} inputs([[SOFTMAX]] : tensor<1x8x4x32xf16>) -> tensor<1x2x16x32xf16>
    // CHECK: return [[SHAPECAST_OUT]] : tensor<1x2x16x32xf16>
}
