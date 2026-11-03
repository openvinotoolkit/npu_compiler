//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true enable-se-ptrs-operations=true" --convert-d2s-to-transposed-conv %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// -----

// CHECK-LABEL: @DepthToSpaceWithConversionBS2
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x3x3xf16>)
func.func @DepthToSpaceWithConversionBS2(%input: tensor<1x64x3x3xf16>) -> tensor<1x16x6x6xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 2 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x64x3x3xf16> -> tensor<1x16x6x6xf16>

    return %d2s : tensor<1x16x6x6xf16>

    // CHECK:                [[WEIGHTS:%.+]] = const.Declare tensor<16x64x2x2xf16>

    // CHECK:                [[INPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:                [[OUTPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:                [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[INPUTRANGE]], [[OUTPUTRANGE]], [[INPUTRANGE]], [[OUTPUTRANGE]]) {
    // CHECK-SAME:               auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:               levels_or_low_fp_type = 2 : i64
    // CHECK-SAME:           }

    // CHECK:                [[TRANSCONV:%.+]] = IE.TransposedConvolution([[INPUT]], [[FAKEQUANTIZE]]) {
    // CHECK-SAME:               dilations = [1, 1],
    // CHECK-SAME:               operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:               pads_begin = [0, 0],
    // CHECK-SAME:               pads_end = [0, 0],
    // CHECK-SAME:               spatial_output_padding = [0, 0],
    // CHECK-SAME:               strides = [2, 2]
    // CHECK-SAME:           } : tensor<1x64x3x3xf16>, tensor<16x64x2x2xf16> -> tensor<1x16x6x6xf16>

    // CHECK:                return [[TRANSCONV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceWithConversionBS3
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x144x3x3xf16>)
func.func @DepthToSpaceWithConversionBS3(%input: tensor<1x144x3x3xf16>) -> tensor<1x16x9x9xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 3 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x144x3x3xf16> -> tensor<1x16x9x9xf16>

    return %d2s : tensor<1x16x9x9xf16>

    // CHECK:                [[WEIGHTS:%.+]]  = const.Declare tensor<16x144x3x3xf16>

    // CHECK:                [[INPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:                [[OUTPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:                [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[INPUTRANGE]], [[OUTPUTRANGE]], [[INPUTRANGE]], [[OUTPUTRANGE]]) {
    // CHECK-SAME:               auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:               levels_or_low_fp_type = 2 : i64
    // CHECK-SAME:           }

    // CHECK:                [[TRANSCONV:%.+]] = IE.TransposedConvolution([[INPUT]], [[FAKEQUANTIZE]]) {
    // CHECK-SAME:               dilations = [1, 1],
    // CHECK-SAME:               operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:               pads_begin = [0, 0],
    // CHECK-SAME:               pads_end = [0, 0],
    // CHECK-SAME:               spatial_output_padding = [0, 0],
    // CHECK-SAME:               strides = [3, 3]
    // CHECK-SAME:           } : tensor<1x144x3x3xf16>, tensor<16x144x3x3xf16> -> tensor<1x16x9x9xf16>

    // CHECK:                return [[TRANSCONV]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthToSpaceWithoutConversionBS4
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x180x270xf16>)
func.func @DepthToSpaceWithoutConversionBS4(%input : tensor<1x16x180x270xf16>) -> tensor<1x1x720x1080xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 4 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x16x180x270xf16> -> tensor<1x1x720x1080xf16>

    return %d2s : tensor<1x1x720x1080xf16>

    // CHECK:                [[D2S:%.+]] = IE.DepthToSpace([[INPUT]]) {
    // CHECK-SAME:               block_size = 4 : i64,
    // CHECK-SAME:               mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    // CHECK-SAME:           } : tensor<1x16x180x270xf16> -> tensor<1x1x720x1080xf16>

    // CHECK:                return [[D2S]] : tensor<1x1x720x1080xf16>
}
// -----

// CHECK-LABEL: @DepthToSpaceWithConversionBS2_SmallChan
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x12x1280x720xf16>)
func.func @DepthToSpaceWithConversionBS2_SmallChan(%input: tensor<1x12x1280x720xf16>) -> tensor<1x3x2560x1440xf16> {
    %d2s = IE.DepthToSpace(%input) {
        block_size = 2 : i64,
        mode = #IE.depth_to_space_mode<DEPTH_FIRST>
    } : tensor<1x12x1280x720xf16> -> tensor<1x3x2560x1440xf16>

    return %d2s : tensor<1x3x2560x1440xf16>

    // CHECK:                [[WEIGHTS:%.+]] = const.Declare tensor<3x12x2x2xf16>

    // CHECK:                [[INPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:                [[OUTPUTRANGE:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    // CHECK:                [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[INPUTRANGE]], [[OUTPUTRANGE]], [[INPUTRANGE]], [[OUTPUTRANGE]]) {
    // CHECK-SAME:               auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:               levels_or_low_fp_type = 2 : i64
    // CHECK-SAME:           }

    // CHECK:                [[TRANSCONV:%.+]] = IE.TransposedConvolution([[INPUT]], [[FAKEQUANTIZE]]) {
    // CHECK-SAME:               dilations = [1, 1],
    // CHECK-SAME:               operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    // CHECK-SAME:               pads_begin = [0, 0],
    // CHECK-SAME:               pads_end = [0, 0],
    // CHECK-SAME:               spatial_output_padding = [0, 0],
    // CHECK-SAME:               strides = [2, 2]
    // CHECK-SAME:           } : tensor<1x12x1280x720xf16>, tensor<3x12x2x2xf16> -> tensor<1x3x2560x1440xf16>

    // CHECK:                return [[TRANSCONV]] : tensor<1x3x2560x1440xf16>
}
