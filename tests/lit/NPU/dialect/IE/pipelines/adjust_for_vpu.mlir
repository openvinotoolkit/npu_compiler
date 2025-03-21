//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-for-vpu %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK: func.func @ConvertNonDepthWiseGroupTransposedConvToConv([[ARG0:%.+]]: tensor<1x32x64x64xf16>) -> tensor<1x32x128x128xf16> {
func.func @ConvertNonDepthWiseGroupTransposedConvToConv(%arg0: tensor<1x32x64x64xf16>) -> tensor<1x32x128x128xf16> {
    %FILTERS = const.Declare tensor<2x16x16x4x4xf16> = dense<1.000000e+00> : tensor<2x16x16x4x4xf16>

    %RESULT = IE.GroupTransposedConvolution(%arg0, %FILTERS) {dilations = [1, 1], output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x32x64x64xf16>, tensor<2x16x16x4x4xf16> -> tensor<1x32x128x128xf16>
    return %RESULT : tensor<1x32x128x128xf16>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x32x4x4xf16> = dense
    // CHECK:           [[UPSAMPLE:%.+]] = IE.Upsampling([[ARG0]])
    // CHECK-SAME:          {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 1]>, upsampling_factor = [2, 2, 1]} : tensor<1x32x64x64xf16> -> tensor<1x32x128x128xf16>
    // CHECK:           [[CONV:%.+]] = IE.Convolution([[UPSAMPLE]], [[WEIGHTS]])
    // CHECK-SAME:          {dilations = [1, 1], pads_begin = [2, 2], pads_end = [1, 1], strides = [1, 1]} : tensor<1x32x128x128xf16>, tensor<32x32x4x4xf16> -> tensor<1x32x128x128xf16>

    // CHECK:           return [[CONV]]
}

// -----

// CHECK-LABEL: @FusePostOpReluIntoConv
func.func @FusePostOpReluIntoConv(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x64x64xf16> {
    %cts = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64x64xf16> -> tensor<1x1x64x64xf16>

    return %1 : tensor<1x1x64x64xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:    post_op = #IE.PostOp<name = "IE.ReLU", attrs = {}>, strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    // CHECK: return [[VAL1]] : tensor<1x1x64x64xf16>
}

// -----

// CHECK-LABEL: @NearestWithSIZESModeConvertToTileOp
func.func @NearestWithSIZESModeConvertToTileOp(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT:   IE.Interpolate
    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[TILE]] : tensor<1x96x33x33xf32>
}

// -----

// CHECK-LABEL: @NearestWithSCALESModeConvertToTileOp
func.func @NearestWithSCALESModeConvertToTileOp(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT:   IE.Interpolate
    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[TILE]] : tensor<1x96x33x33xf32>
}
