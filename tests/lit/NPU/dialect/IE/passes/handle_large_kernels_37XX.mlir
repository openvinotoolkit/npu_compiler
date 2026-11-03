//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: platform-NPU3720

// The DMA max planes limit is 256 on NPU3720 only. For NHWC->NHWC depth-first SpaceToDepth
// numPlanes equals the input channel count, so IC above the limit must not be folded.

// CHECK-LABEL: @SpaceToDepthLargeConvInputChannelsExceedDmaPlanes
func.func @SpaceToDepthLargeConvInputChannelsExceedDmaPlanes(%arg0: tensor<1x272x32x32xf16>) -> tensor<1x4x2x2xf16> {
    %cst = const.Declare tensor<4x272x16x16xf16> = dense<1.000000e+00> : tensor<4x272x16x16xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x272x32x32xf16>, tensor<4x272x16x16xf16> -> tensor<1x4x2x2xf16>

    return %conv : tensor<1x4x2x2xf16>

    // CHECK-NOT: IE.SpaceToDepthOp
}

// -----

// CHECK-LABEL: @SpaceToDepthLargeConvInputChannelsWithinDmaPlanes
func.func @SpaceToDepthLargeConvInputChannelsWithinDmaPlanes(%arg0: tensor<1x3x32x32xf16>) -> tensor<1x4x2x2xf16> {
    %cst = const.Declare tensor<4x3x16x16xf16> = dense<1.000000e+00> : tensor<4x3x16x16xf16>
    %conv = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [16, 16]} : tensor<1x3x32x32xf16>, tensor<4x3x16x16xf16> -> tensor<1x4x2x2xf16>

    return %conv : tensor<1x4x2x2xf16>

    // CHECK: IE.SpaceToDepthOp
}
