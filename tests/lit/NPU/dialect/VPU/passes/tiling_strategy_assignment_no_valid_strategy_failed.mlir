//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: not vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A CMX budget of 1 byte cannot fit any tile, no matter how the op is split, so the tiling
// search is expected to hard-fail instead of silently leaving the op untiled.
config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1 bytes of @CMX_NN
}

// CHECK: error: 'VPU.NCE.MaxPool' op failed to find a tiling strategy that fits the operation into CMX
func.func @NoValidTilingStrategy(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x4x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A regular SW op still requires all buffers to fit into CMX when left untiled.
config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1 bytes of @CMX_NN
}

// CHECK: error: 'VPU.Add' op failed to find a tiling strategy that fits the operation into CMX
func.func @NoValidTilingStrategySW(%arg0: tensor<1x16x4x4xf16, {order = #NHWC}>,
                                  %arg1: tensor<1x16x4x4xf16, {order = #NHWC}>)
        -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x16x4x4xf16, {order = #NHWC}>, tensor<1x16x4x4xf16, {order = #NHWC}> ->
            tensor<1x16x4x4xf16, {order = #NHWC}>
    return %0 : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#bounds = #const.OpaqueI64Elements<[1, 16, 4, 4]> : tensor<4xsi64>

// Dynamic ops are skipped by the CMX-fit hard-failure check until E#229192 is fixed.
// CHECK-NOT: error: 'VPU.Convert' op failed to find a tiling strategy that fits the operation into CMX
func.func @DynamicNoValidTilingStrategy(%arg0: tensor<1x16x?x?xf32, {bounds = #bounds, order = #NHWC}>)
        -> tensor<1x16x?x?xf16, {bounds = #bounds, order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x16x?x?xf32, {bounds = #bounds, order = #NHWC}> ->
        tensor<1x16x?x?xf16, {bounds = #bounds, order = #NHWC}>
    return %0 : tensor<1x16x?x?xf16, {bounds = #bounds, order = #NHWC}>
}
