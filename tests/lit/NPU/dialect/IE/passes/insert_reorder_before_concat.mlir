//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --layer-reorder-concat-pass --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK: func.func @InsertReorderBeforeConcat(%arg0: tensor<1x8x512x64xf16>, %arg1: tensor<1x2x1x512xf16>) -> tensor<1x64x9x512xf16>

func.func @InsertReorderBeforeConcat(%arg0: tensor<1x8x512x64xf16>, %arg1: tensor<1x2x1x512xf16>) -> tensor<1x64x9x512xf16> {
    %cst = const.Declare tensor<64x2x1x1xf16> = dense<1.0>
        : tensor<64x2x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Transpose(%arg0) {order_value = #NWCH}
        : tensor<1x8x512x64xf16> -> tensor<1x64x8x512xf16>

    %1 = IE.Convolution(%arg1, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x2x1x512xf16>, tensor<64x2x1x1xf16> -> tensor<1x64x1x512xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    } : tensor<1x64x8x512xf16>, tensor<1x64x1x512xf16> -> tensor<1x64x9x512xf16>

    return %2 : tensor<1x64x9x512xf16>

    // CHECK-DAG:   %[[CONSTANT_1:.*]] = const.Declare tensor<64x2x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<64x2x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:   %[[TRANSPOSE:.*]] = IE.Transpose(%arg0) {order_value = #NWCH}
    // CHECK-SAME   : tensor<1x8x512x64xf16> -> tensor<1x64x8x512xf16>

    // CHECK:   %[[CONV2D:.*]] = IE.Convolution(%arg1, %[[CONSTANT_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x2x1x512xf16>, tensor<64x2x1x1xf16> -> tensor<1x64x1x512xf16>

    // CHECK:   %[[TRANSPOSE_NHWC:.*]] = IE.Reorder(%[[TRANSPOSE]]) {dstOrder = #NHWC}
    // CHECK-SAME:  : tensor<1x64x8x512xf16> -> tensor<1x64x8x512xf16, {order = #NHWC}>
    // CHECK:   %[[CONV2D_NHWC:.*]] = IE.Reorder(%[[CONV2D]]) {dstOrder = #NHWC}
    // CHECK-SAME:  : tensor<1x64x1x512xf16> -> tensor<1x64x1x512xf16, {order = #NHWC}>

    // CHECK:   %[[CONCAT:.*]] = IE.Concat(%[[TRANSPOSE_NHWC]], %[[CONV2D_NHWC]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    // CHECK-SAME:  } : tensor<1x64x8x512xf16, {order = #NHWC}>, tensor<1x64x1x512xf16, {order = #NHWC}> -> tensor<1x64x9x512xf16, {order = #NHWC}>

    // CHECK:   %[[REORDER:.*]] = IE.Reorder(%[[CONCAT]]) {dstOrder = #NCHW}
    // CHECK-SAME:  : tensor<1x64x9x512xf16, {order = #NHWC}> -> tensor<1x64x9x512xf16>

    // CHECK:   return %[[REORDER]] : tensor<1x64x9x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @InsertReorderBeforeReshapeConcat(%arg0: tensor<1x8x512x64xf16>, %arg1: tensor<1x2x1x512xf16>) -> tensor<1x64x9x512xf16>

func.func @InsertReorderBeforeReshapeConcat(%arg0: tensor<1x8x512x64xf16>, %arg1: tensor<1x2x1x512xf16>) -> tensor<1x64x9x512xf16> {
    %cst = const.Declare tensor<64x2x1x1xf16> = dense<1.0>
        : tensor<64x2x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2], [2], [3], [3]], shape_value = [1, 64, 8, 512]}
        : tensor<1x8x512x64xf16> -> tensor<1x64x8x512xf16>

    %1 = IE.Convolution(%arg1, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x2x1x512xf16>, tensor<64x2x1x1xf16> -> tensor<1x64x1x512xf16>

    %2 = IE.Concat(%0, %1) {
        static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    } : tensor<1x64x8x512xf16>, tensor<1x64x1x512xf16> -> tensor<1x64x9x512xf16>

    return %2 : tensor<1x64x9x512xf16>

    // CHECK-DAG:   %[[CONSTANT_1:.*]] = const.Declare tensor<64x2x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:  : tensor<64x2x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:   %[[RESHAPE:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME   : tensor<1x8x512x64xf16> -> tensor<1x64x8x512xf16>

    // CHECK:   %[[CONV2D:.*]] = IE.Convolution(%arg1, %[[CONSTANT_1]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x2x1x512xf16>, tensor<64x2x1x1xf16> -> tensor<1x64x1x512xf16>

    // CHECK:   %[[RESHAPE_NHWC:.*]] = IE.Reorder(%[[RESHAPE]]) {dstOrder = #NHWC}
    // CHECK-SAME:  : tensor<1x64x8x512xf16> -> tensor<1x64x8x512xf16, {order = #NHWC}>
    // CHECK:   %[[CONV2D_NHWC:.*]] = IE.Reorder(%[[CONV2D]]) {dstOrder = #NHWC}
    // CHECK-SAME:  : tensor<1x64x1x512xf16> -> tensor<1x64x1x512xf16, {order = #NHWC}>

    // CHECK:   %[[CONCAT:.*]] = IE.Concat(%[[RESHAPE_NHWC]], %[[CONV2D_NHWC]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]]
    // CHECK-SAME:  } : tensor<1x64x8x512xf16, {order = #NHWC}>, tensor<1x64x1x512xf16, {order = #NHWC}> -> tensor<1x64x9x512xf16, {order = #NHWC}>

    // CHECK:   %[[REORDER:.*]] = IE.Reorder(%[[CONCAT]]) {dstOrder = #NCHW}
    // CHECK-SAME:  : tensor<1x64x9x512xf16, {order = #NHWC}> -> tensor<1x64x9x512xf16>

    // CHECK:   return %[[REORDER]] : tensor<1x64x9x512xf16>
}
