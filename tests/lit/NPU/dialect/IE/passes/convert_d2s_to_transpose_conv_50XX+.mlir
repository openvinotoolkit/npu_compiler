//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-d2s-to-transposed-conv %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceWithConversionBS2
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x3x3xf16>)
func.func @DepthToSpaceWithConversionBS2(%input: tensor<1x4x3x3xf16>) -> tensor<1x1x6x6xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 2 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x4x3x3xf16> -> tensor<1x1x6x6xf16>

    return %d2s : tensor<1x1x6x6xf16>

    // CHECK:                [[WEIGHTS:%.+]] = const.Declare tensor<1x4x2x2xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[0, 0], [0, 1]],
    // CHECK-SAME{LITERAL}:              [[0, 0], [1, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 1], [0, 0]],
    // CHECK-SAME{LITERAL}:              [[1, 0], [0, 0]]]]> :
    // CHECK-SAME:               tensor<1x4x2x2xui8, {order = #NHWC}>,
    // CHECK-SAME:               [#const.CastElemType<f16>, #const.Reorder<#NCHW>]

    // CHECK:                [[INPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:                [[OUTPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:                [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[INPUTRANGE]], [[OUTPUTRANGE]], [[INPUTRANGE]], [[OUTPUTRANGE]]) {
    // CHECK-SAME:               auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:               levels = 2 : i64
    // CHECK-SAME:           }

    // CHECK:                [[TRANSCONV:%.+]] = IE.TransposedConvolution([[INPUT]], [[FAKEQUANTIZE]]) {
    // CHECK-SAME:               dilations = [1, 1],
    // CHECK-SAME:               operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:               pads_begin = [0, 0],
    // CHECK-SAME:               pads_end = [0, 0],
    // CHECK-SAME:               spatial_output_padding = [0, 0],
    // CHECK-SAME:               strides = [2, 2]
    // CHECK-SAME:           } : tensor<1x4x3x3xf16>, tensor<1x4x2x2xf16> -> tensor<1x1x6x6xf16>

    // CHECK:                return [[TRANSCONV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceWithConversionBS3
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x9x3x3xf16>)
func.func @DepthToSpaceWithConversionBS3(%input: tensor<1x9x3x3xf16>) -> tensor<1x1x9x9xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 3 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x9x3x3xf16> -> tensor<1x1x9x9xf16>

    return %d2s : tensor<1x1x9x9xf16>

    // CHECK:                [[WEIGHTS:%.+]]  = const.Declare tensor<1x9x3x3xf16> =
    // CHECK-SAME{LITERAL}:      dense<[[[[0, 0, 0], [0, 0, 0], [0, 0, 1]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
    // CHECK-SAME{LITERAL}:              [[1, 0, 0], [0, 0, 0], [0, 0, 0]]]]> :
    // CHECK-SAME:               tensor<1x9x3x3xui8, {order = #NHWC}>,
    // CHECK-SAME:               [#const.CastElemType<f16>, #const.Reorder<#NCHW>]

    // CHECK:                [[INPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:                [[OUTPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:                [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[INPUTRANGE]], [[OUTPUTRANGE]], [[INPUTRANGE]], [[OUTPUTRANGE]]) {
    // CHECK-SAME:               auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:               levels = 2 : i64
    // CHECK-SAME:           }

    // CHECK:                [[TRANSCONV:%.+]] = IE.TransposedConvolution([[INPUT]], [[FAKEQUANTIZE]]) {
    // CHECK-SAME:               dilations = [1, 1],
    // CHECK-SAME:               operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:               pads_begin = [0, 0],
    // CHECK-SAME:               pads_end = [0, 0],
    // CHECK-SAME:               spatial_output_padding = [0, 0],
    // CHECK-SAME:               strides = [3, 3]
    // CHECK-SAME:           } : tensor<1x9x3x3xf16>, tensor<1x9x3x3xf16> -> tensor<1x1x9x9xf16>

    // CHECK:                return [[TRANSCONV]]
}
