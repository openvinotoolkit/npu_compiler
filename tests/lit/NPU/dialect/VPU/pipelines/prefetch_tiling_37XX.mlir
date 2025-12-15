//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --tiling="prefetching=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: func.func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @SplitSwConvOverOC(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE0:%.+]] = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE1:%.+]] = VPU.Slice [[BIAS]] [0, 128, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x244xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x244x244xf16>)
            -> tensor<1x16x244x244xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x16x244x244xf16> -> tensor<1x16x244x244xf16>
    return %1 : tensor<1x16x244x244xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 123, 244]
    // CHECK-SAME:       : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 121, 0] [1, 16, 123, 244]
    // CHECK-SAME:      : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 122, 0]
    // CHECK-SAME:      -> tensor<1x16x244x244xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x244xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSoftMaxOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func.func @SplitSoftMaxOverH(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftMax([[INPUT_TILE0]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 43, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftMax([[INPUT_TILE1]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 86, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.SoftMax([[INPUT_TILE2]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 129, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.SoftMax([[INPUT_TILE3]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 172, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.SoftMax([[INPUT_TILE4]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 214, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.SoftMax([[INPUT_TILE5]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]
    // CHECK-SAME:      -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

// CHECK-LABEL: func.func @SplitLogSoftmaxOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func.func @SplitLogSoftmaxOverH(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.LogSoftmax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogSoftmax([[INPUT_TILE0]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 43, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogSoftmax([[INPUT_TILE1]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 86, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.LogSoftmax([[INPUT_TILE2]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 129, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.LogSoftmax([[INPUT_TILE3]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 172, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.LogSoftmax([[INPUT_TILE4]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 214, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.LogSoftmax([[INPUT_TILE5]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]
    // CHECK-SAME:      -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x24x64x64xf16>)
            -> tensor<1x24x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
        tensor<1x24x64x64xf16> -> tensor<1x24x256x256xf16>

    return %0 : tensor<1x24x256x256xf16>
}

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 24, 33, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x33x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x24x33x64xf16>
// CHECK-SAME:      -> tensor<1x24x128x256xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 31, 0] [1, 24, 33, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x33x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x24x33x64xf16>
// CHECK-SAME:      -> tensor<1x24x128x256xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 128, 0]]}
// CHECK-SAME:      : tensor<1x24x128x256xf16>, tensor<1x24x128x256xf16> -> tensor<1x24x256x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCDueToUglyScalingFactor
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x21x65x65xf16, {order = #NHWC}>
func.func @InterpSplitOverCDueToUglyScalingFactor(%arg0: tensor<1x21x65x65xf16, {order = #NHWC}>) -> tensor<1x21x513x513xf16, {order = #NHWC}> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode =  <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
    axes_attr = [2, 3],
    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    scales_attr = [7.8923077583312988, 7.8923077583312988],
    sizes_attr = [513, 513]} : tensor<1x21x65x65xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x21x513x513xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  return %0 : tensor<1x21x513x513xf16, {order = #NHWC}>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 21, 12, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x12x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x12x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x86x513xf16, {order = #NHWC}>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 10, 0] [1, 21, 13, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x86x513xf16, {order = #NHWC}>
// CHECK:       [[TILE2:%.+]] = VPU.Slice %arg0 [0, 0, 21, 0] [1, 21, 13, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x86x513xf16, {order = #NHWC}>
// CHECK:       [[TILE3:%.+]] = VPU.Slice %arg0 [0, 0, 32, 0] [1, 21, 13, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP3:%.+]] = VPU.Interpolate([[TILE3]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x85x513xf16, {order = #NHWC}>
// CHECK:       [[TILE4:%.+]] = VPU.Slice %arg0 [0, 0, 43, 0] [1, 21, 13, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP4:%.+]] = VPU.Interpolate([[TILE4]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x13x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x85x513xf16, {order = #NHWC}>
// CHECK:       [[TILE5:%.+]] = VPU.Slice %arg0 [0, 0, 54, 0] [1, 21, 11, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x21x11x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP5:%.+]] = VPU.Interpolate([[TILE5]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x11x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x21x85x513xf16, {order = #NHWC}>
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]], [[INTERP3]], [[INTERP4]], [[INTERP5]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 172, 0], [0, 0, 258, 0], [0, 0, 343, 0], [0, 0, 428, 0]]} : tensor<1x21x86x513xf16, {order = #NHWC}>, tensor<1x21x86x513xf16, {order = #NHWC}>, tensor<1x21x86x513xf16, {order = #NHWC}>, tensor<1x21x85x513xf16, {order = #NHWC}>, tensor<1x21x85x513xf16, {order = #NHWC}>, tensor<1x21x85x513xf16, {order = #NHWC}> -> tensor<1x21x513x513xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x21x513x513xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCDueTo1Size
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @InterpSplitOverCDueTo1Size(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x65x65xf16, {order = #NHWC}> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode =  <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
    axes_attr = [2, 3],
    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    scales_attr = [6.500000e+01, 6.500000e+01],
    sizes_attr = [65, 65]} : tensor<1x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x65x65xf16, {order = #NHWC}>

    return %0 : tensor<1x256x65x65xf16, {order = #NHWC}>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 1, 1]
// CHECK-SAME:      : tensor<1x256x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
// CHECK-SAME:      : tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x128x65x65xf16, {order = #NHWC}>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 128, 0, 0] [1, 128, 1, 1]
// CHECK-SAME:      : tensor<1x256x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
// CHECK-SAME:      : tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x128x65x65xf16, {order = #NHWC}>
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x65x65xf16, {order = #NHWC}>, tensor<1x128x65x65xf16, {order = #NHWC}> -> tensor<1x256x65x65xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x256x65x65xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @SplitPReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitPReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x1280xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.PRelu([[INPUT_TILE0]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640]  [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.PRelu([[INPUT_TILE1]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


  }

// -----

// CHECK-LABEL: func.func @SplitLeakyReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LeakyRelu([[INPUT_TILE0]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LeakyRelu([[INPUT_TILE1]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

  }


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<256x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<256x1x1x4xsi32, {order = #NHWC}>
func.func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<256x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<256x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x256x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 144, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<256x144x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32, {order = #NHWC}> -> tensor<1x256x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x256x20x20xf16, {order = #NHWC}>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>
    // CHECK:        [[ADD:%.+]] = VPU.NCE.Eltwise([[CONV]], [[CONV]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>
    // CHECK:        [[CONV1:%.+]] = VPU.NCE.Convolution([[ADD]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x256x20x20xf16, {order = #NHWC}>
    // CHECK:        return [[CONV1]] : tensor<1x256x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x340x340xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 48, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 97, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 146, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 195, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 243, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 291, 0] [1, 16, 49, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE6]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 49, 0], [0, 0, 98, 0], [0, 0, 147, 0], [0, 0, 196, 0], [0, 0, 244, 0], [0, 0, 292, 0]
    // CHECK-SAME:      -> tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x340xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverHAndKeepWidthSize
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverHAndKeepWidthSize(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x170x170xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x170x170xf16, {order = #NHWC}>

    return %0 : tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 68, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 136, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 204, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 272, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 34, 0], [0, 0, 68, 0], [0, 0, 102, 0], [0, 0, 136, 0]
    // CHECK-SAME:      -> tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x170x170xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverWAndKeepHeightSize
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x2x80000xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverWAndKeepHeightSize(%arg0: tensor<1x16x2x80000xf16, {order = #NHWC}>) -> tensor<1x16x1x40000xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x1x40000xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x40000xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 2, 11429]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11429xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5715xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 11430] [1, 16, 2, 11429]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11429xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5715xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 22860] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 34288] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 45716] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 57144] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 68572] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE6]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 5715], [0, 0, 0, 11430], [0, 0, 0, 17144], [0, 0, 0, 22858], [0, 0, 0, 28572], [0, 0, 0, 34286]
    // CHECK-SAME:      -> tensor<1x16x1x40000xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x40000xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func.func @NoTileWithSOH(
        %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
            -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1>
        : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x100x100xf16, {order = #NHWC}>, tensor<128x32x3x3xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x210xf16, {order = #NHWC}>
func.func @TileWithSOH(
        %arg0: tensor<1x16x210x210xf16, {order = #NHWC}>)
            -> tensor<1x32x210x210xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
        : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x16x210x210xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x210x210xf16, {order = #NHWC}>

    return %0 : tensor<1x32x210x210xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 104, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x32x210x210xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
func.func @NoTileWithSOK(
        %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
            -> tensor<1x240x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<240x1x1x4xsi32> = dense<1>
        : tensor<240x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<240x32x7x7xf16, {order = #NHWC}>, tensor<240x1x1x4xsi32> -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func.func @TileWithSOK(
        %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
            -> tensor<1x768x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1>
        : tensor<768x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [768, 32, 7, 7],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x30x30xf16, {order = #NHWC}>, tensor<768x32x7x7xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:         [[CONST:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    // CHECK:         [[CONST0:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
    // CHECK:         [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 11, 30] : tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x11x30xf16, {order = #NHWC}>
    // CHECK:         [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[CONST]], [[CONST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
    // CHECK-SAME:         -> tensor<1x768x8x30xf16, {order = #NHWC}>
    // CHECK:         [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 5, 0] [1, 32, 14, 30] : tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x14x30xf16, {order = #NHWC}>
    // CHECK:         [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[CONST]], [[CONST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
    // CHECK-SAME:         -> tensor<1x768x8x30xf16, {order = #NHWC}>
    // CHECK:         [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 13, 0] [1, 32, 13, 30] : tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x13x30xf16, {order = #NHWC}>
    // CHECK:         [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[CONST]], [[CONST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
    // CHECK-SAME:         -> tensor<1x768x7x30xf16, {order = #NHWC}>
    // CHECK:         [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 32, 10, 30] : tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x10x30xf16, {order = #NHWC}>
    // CHECK:         [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[CONST]], [[CONST0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 0 : i64, bottom = 3 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
    // CHECK-SAME:         -> tensor<1x768x7x30xf16, {order = #NHWC}>
    // CHECK:         [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 23, 0]]} :
    // CHECK-SAME:        tensor<1x768x8x30xf16, {order = #NHWC}>, tensor<1x768x8x30xf16, {order = #NHWC}>, tensor<1x768x7x30xf16, {order = #NHWC}>, tensor<1x768x7x30xf16, {order = #NHWC}> -> tensor<1x768x30x30xf16, {order = #NHWC}>
    // CHECK:         return [[CONCAT]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOKFor
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func.func @LargeConstPipeliningSOKFor(
        %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
            -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1>
        : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>, [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>, [#const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>,  [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [256, 256, 3, 3]>, #const.Reorder<#NHWC>]

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
func.func @SplitNCEEltwise(
        %arg0: tensor<1x512x28x28xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x28x28xf16, {order = #NHWC}>)
            -> tensor<1x512x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x512x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x512x28x28xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[INPUT_0_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0_0]], [[INPUT_1_0]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[INPUT_0_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[INPUT_0_1]], [[INPUT_1_1]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Concat
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      : tensor<1x256x28x28xf16, {order = #NHWC}>, tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x512x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x70xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x70xf16, {order = #NHWC}>
func.func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x70x70xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x70x70xf16, {order = #NHWC}>)
            -> tensor<1x64x70x70xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x70x70xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x70x70xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    return %1 : tensor<1x64x70x70xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK-NOT:   VPU.Slice
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x70x70xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x80x80xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>, tensor<160x1x1x4xsi32, {order = #NCHW}> -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>

    // CHECK-DAG:        [[WEIGHTS_TILE:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>

    // CHECK-DAG:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_SM_TILE]], [[WEIGHTS_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_2]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x11520xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x11520xf16, {order = #NHWC}>) -> tensor<1x16x1x11520xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x11520xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x11520xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2880] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 5760] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 3

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 8640] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE3]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>


    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 2880], [0, 0, 0, 5760], [0, 0, 0, 8640]
    // CHECK-SAME:      -> tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x11520xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16>) -> tensor<1x1x1x368640xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x368640xf16> -> tensor<1x1x1x368640xf16>

    return %0 : tensor<1x1x1x368640xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 122880] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 245760] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.AvgPool([[INPUT_TILE2]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 122880], [0, 0, 0, 245760]
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x8x10001xf16, {order = #NHWC}>
func.func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x8x10001xf16, {order = #NHWC}>) -> tensor<1x16x8x10001xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x8x10001xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x10001xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 8, 3334]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3334xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[SLICE_0]])
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3334xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 3334] [1, 16, 8, 3334]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3334xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[SLICE_1]])
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3334xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 6668] [1, 16, 8, 3333]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3333xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[SLICE_2]])
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3333xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]], [[MAXPOOL_2]])
    // CHECK:       return [[CONCAT]] : tensor<1x16x8x10001xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PrefetchingSpaceToDepthParent
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x512x512xf16, {order = #NHWC}>
func.func @PrefetchingSpaceToDepthParent(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<64x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x48x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x48x128x128xf16, {order = #NHWC}>

    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64, fp_prelu_alpha = 14.226995468139648 : f64,
        rawFilterShape = [64, 48, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEInt<mode = <LPRELU>>
    } : tensor<1x48x128x128xf16, {order = #NHWC}>, tensor<64x48x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x128x128xf16, {order = #NHWC}>

    return %1 : tensor<1x64x128x128xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WT_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<64x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[SL0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 512, 256] : tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x3x512x256xf16, {order = #NHWC}>

    // CHECK:       [[SpaceToDepthOp0:%.+]] = VPU.SpaceToDepthOp([[SL0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
    // CHECK-SAME:      : tensor<1x3x512x256xf16, {order = #NHWC}> -> tensor<1x48x128x64xf16, {order = #NHWC}>

    // CHECK:       [[SL1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 256] [1, 3, 512, 256] : tensor<1x3x512x512xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x3x512x256xf16, {order = #NHWC}>

    // CHECK:       [[SpaceToDepthOp1:%.+]] = VPU.SpaceToDepthOp([[SL1]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>}
    // CHECK-SAME:      : tensor<1x3x512x256xf16, {order = #NHWC}> -> tensor<1x48x128x64xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT0:%.+]] = VPU.Concat([[SpaceToDepthOp0]], [[SpaceToDepthOp1]])

    // CHECK:       [[SL2:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 48, 33, 128] : tensor<1x48x128x128xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x48x33x128xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[SL2]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 14.226995468139648 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64,
    // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <LPRELU>>,
    // CHECK-SAME:      rawFilterShape = [64, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SL3:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 31, 0] [1, 48, 34, 128] : tensor<1x48x128x128xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x48x34x128xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SL3]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 14.226995468139648 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64,
    // CHECK-SAME:      rawFilterShape = [64, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SL4:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 63, 0] [1, 48, 34, 128] : tensor<1x48x128x128xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x48x34x128xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SL4]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 14.226995468139648 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64,
    // CHECK-SAME:      rawFilterShape = [64, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SL5:%.+]] = VPU.Slice [[CONCAT0]] [0, 0, 95, 0] [1, 48, 33, 128] : tensor<1x48x128x128xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x48x33x128xf16, {order = #NHWC}>

    // Tile 3
    // CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[SL5]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 14.226995468139648 : f64, lrelu_mult = 1229 : i64, lrelu_shift = 12 : i64,
    // CHECK-SAME:      rawFilterShape = [64, 48, 3, 3], strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x32x128xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT2:%.+]] = VPU.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]])
    // CHECK:       return [[CONCAT2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PrefetchingNCEToSWTask
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x80x80xf16, {order = #NHWC}>
func.func @PrefetchingNCEToSWTask(%arg0: tensor<1x64x80x80xf16, {order = #NHWC}>) -> tensor<1x48x80x80xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<48x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

    %0 = VPU.Tanh(%arg0) : tensor<1x64x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        rawFilterShape = [48, 64, 5, 5],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x64x80x80xf16, {order = #NHWC}>, tensor<48x64x5x5xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x80x80xf16, {order = #NHWC}>

    return %1 : tensor<1x48x80x80xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WT_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_2:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WT_1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WT_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[TANH:%.+]] = VPU.Tanh([[INPUT]])
    // CHECK-SAME:      : tensor<1x64x80x80xf16, {order = #NHWC}> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[TANH]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[TANH]], [[WEIGHTS_1]], [[WT_1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[TANH]], [[WEIGHTS_2]], [[WT_2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV]], [[CONV1]], [[CONV2]])
    // CHECK:       return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:    max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:    max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @LogSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Log(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Log([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Log([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @AbsSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @AbsSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Abs(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Abs([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Abs([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitFloorModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.FloorMod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.FloorMod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Mod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Mod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Mod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitPowerEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Power([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Power([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalOrEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalOr([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalOr([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalXorEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalXor([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalXor([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
  %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
  return %0 : tensor<1x10x256x256xi8>
}

// CHECK-LABEL: @SplitEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Equal([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Equal([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xi8>, tensor<1x10x128x256xi8> -> tensor<1x10x256x256xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>

// -----

func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
  %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
  return %0 : tensor<1x10x256x256xi8>
}

// CHECK-LABEL: @SplitNotEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NotEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NotEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xi8>, tensor<1x10x128x256xi8> -> tensor<1x10x256x256xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>

// -----

func.func @SplitSoftPlusActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.SoftPlus(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSoftPlusActivationSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftPlus([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftPlus([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
  %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
  return %0 : tensor<1x10x256x256xi8>
}

// CHECK-LABEL: @SplitLessEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Less([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Less([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xi8>, tensor<1x10x128x256xi8> -> tensor<1x10x256x256xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>

// -----

func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
  %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
  return %0 : tensor<1x10x256x256xi8>
}

// CHECK-LABEL: @SplitLessEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LessEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LessEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xi8>, tensor<1x10x128x256xi8> -> tensor<1x10x256x256xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>

// -----

func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
  %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
  return %0 : tensor<1x10x256x256xi8>
}

// CHECK-LABEL: @SplitGreaterEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Greater([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Greater([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xi8>, tensor<1x10x128x256xi8> -> tensor<1x10x256x256xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>

// -----

func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitGreaterEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.GreaterEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.GreaterEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitErfOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitErfOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Erf(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Erf([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Erf([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SplitFloorOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitFloorOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Floor(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Floor([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Floor([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @TanSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tan(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tan([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tan([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SwishSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Swish([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Swish([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @HSigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSigmoid([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSigmoid([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SplitNegativeActivationSw
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Negative(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Negative([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Negative([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Ceiling([[INPUT_TILE0]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Ceiling([[INPUT_TILE1]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sign(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSignActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sign([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sign([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>, %arg2: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitSelectEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Select([[INPUT_TILE0]], [[INPUT_TILE1]], [[INPUT_TILE2]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_0]] [0, 2, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT_1]] [0, 2, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT_2]] [0, 2, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Select([[INPUT_TILE3]], [[INPUT_TILE4]], [[INPUT_TILE5]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT_0]] [0, 4, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT_1]] [0, 4, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT_2]] [0, 4, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Select([[INPUT_TILE6]], [[INPUT_TILE7]], [[INPUT_TILE8]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE9:%.+]] = VPU.Slice [[INPUT_0]] [0, 6, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE10:%.+]] = VPU.Slice [[INPUT_1]] [0, 6, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE11:%.+]] = VPU.Slice [[INPUT_2]] [0, 6, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.Select([[INPUT_TILE9]], [[INPUT_TILE10]], [[INPUT_TILE11]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x256xf16>
// CHECK:       [[INPUT_TILE12:%.+]] = VPU.Slice [[INPUT_0]] [0, 8, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE13:%.+]] = VPU.Slice [[INPUT_1]] [0, 8, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[INPUT_TILE14:%.+]] = VPU.Slice [[INPUT_2]] [0, 8, 0, 0] [1, 2, 256, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.Select([[INPUT_TILE12]], [[INPUT_TILE13]], [[INPUT_TILE14]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x2x256x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0], [0, 6, 0, 0], [0, 8, 0, 0]
// CHECK-SAME:  : tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16>, tensor<1x2x256x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitAddEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitAddEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Round([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Round([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitGeluActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Gelu([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Gelu([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


// -----

func.func @SplitReduceSum(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x368x480xf16> -> tensor<1x1x368x480xf16>
  return %0 : tensor<1x1x368x480xf16>
    // CHECK-LABEL: @SplitReduceSum
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 12, 368, 160]
    // CHECK-SAME:   : tensor<1x12x368x480xf16> to tensor<1x12x368x160xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReduceSum([[INPUT_TILE0]]) {
    // CHECK-SAME:  axes_value = [1], keep_dims} : tensor<1x12x368x160xf16> -> tensor<1x1x368x160xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 160] [1, 12, 368, 160]
    // CHECK-SAME:   : tensor<1x12x368x480xf16> to tensor<1x12x368x160xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReduceSum([[INPUT_TILE1]]) {
    // CHECK-SAME:  axes_value = [1], keep_dims} : tensor<1x12x368x160xf16> -> tensor<1x1x368x160xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 320] [1, 12, 368, 160]
    // CHECK-SAME:   : tensor<1x12x368x480xf16> to tensor<1x12x368x160xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.ReduceSum([[INPUT_TILE2]]) {
    // CHECK-SAME:  axes_value = [1], keep_dims} : tensor<1x12x368x160xf16> -> tensor<1x1x368x160xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 160], [0, 0, 0, 320]
    // CHECK-SAME:   : tensor<1x1x368x160xf16>, tensor<1x1x368x160xf16>, tensor<1x1x368x160xf16> -> tensor<1x1x368x480xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x368x480xf16>
}

// -----

func.func @SplitTileOnTwoDim(%arg0: tensor<1x1x2880x50xf16>) -> tensor<1x2x8640x150xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [1, 2, 3, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x2x8640x150xf16>
  return %0 : tensor<1x2x8640x150xf16>

    // CHECK-LABEL: @SplitTileOnTwoDim
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x2880x50xf16>) -> tensor<1x2x8640x150xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x1x2880x150xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 2880, 0], [0, 0, 5760, 0], [0, 1, 0, 0], [0, 1, 2880, 0], [0, 1, 5760, 0]
    // CHECK-SAME:   : tensor<1x1x2880x150xf16>, tensor<1x1x2880x150xf16>, tensor<1x1x2880x150xf16>, tensor<1x1x2880x150xf16>, tensor<1x1x2880x150xf16>, tensor<1x1x2880x150xf16> -> tensor<1x2x8640x150xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2x8640x150xf16>
}

// -----

func.func @SplitTileRepeatsNotOne(%arg0: tensor<3x2x723x25xf16>) -> tensor<6x4x2169x75xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [2, 2, 3, 3]} : tensor<3x2x723x25xf16> -> tensor<6x4x2169x75xf16>
  return %0 : tensor<6x4x2169x75xf16>

    // CHECK-LABEL: @SplitTileRepeatsNotOne
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<3x2x723x25xf16>) -> tensor<6x4x2169x75xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT_TILE0]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT_TILE1]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT_TILE2]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.Tile([[INPUT_TILE3]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [2, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.Tile([[INPUT_TILE4]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [2, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.Tile([[INPUT_TILE5]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.Tile([[INPUT_TILE6]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE7:%.+]] = VPU.Tile([[INPUT_TILE7]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE8:%.+]] = VPU.Tile([[INPUT_TILE8]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE9:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE9:%.+]] = VPU.Tile([[INPUT_TILE9]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE10:%.+]] = VPU.Slice [[INPUT]] [2, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE10:%.+]] = VPU.Tile([[INPUT_TILE10]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[INPUT_TILE11:%.+]] = VPU.Slice [[INPUT]] [2, 0, 0, 0] [1, 2, 723, 25] : tensor<3x2x723x25xf16> to tensor<1x2x723x25xf16>
    // CHECK:       [[OUTPUT_TILE11:%.+]] = VPU.Tile([[INPUT_TILE11]]) {repeats_values = [1, 1, 3, 3]} : tensor<1x2x723x25xf16> -> tensor<1x2x2169x75xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]], [[OUTPUT_TILE7]], [[OUTPUT_TILE8]], [[OUTPUT_TILE9]], [[OUTPUT_TILE10]], [[OUTPUT_TILE11]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [2, 0, 0, 0], [2, 2, 0, 0], [3, 0, 0, 0], [3, 2, 0, 0], [4, 0, 0, 0], [4, 2, 0, 0], [5, 0, 0, 0], [5, 2, 0, 0]
    // CHECK-SAME:   : tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16>, tensor<1x2x2169x75xf16> -> tensor<6x4x2169x75xf16>

    // CHECK:       return [[OUTPUT]] : tensor<6x4x2169x75xf16>
}

// -----

func.func @SplitTileRepeatsIsOne(%arg0: tensor<2x3x360x50xf16>) -> tensor<6x3x1080x100xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [3, 1, 3, 2]} : tensor<2x3x360x50xf16> -> tensor<6x3x1080x100xf16>
  return %0 : tensor<6x3x1080x100xf16>

    // CHECK-LABEL: @SplitTileRepeatsIsOne
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<2x3x360x50xf16>) -> tensor<6x3x1080x100xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 3, 2]} : tensor<2x3x360x50xf16> -> tensor<2x3x1080x100xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 3, 2]} : tensor<2x3x360x50xf16> -> tensor<2x3x1080x100xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 3, 2]} : tensor<2x3x360x50xf16> -> tensor<2x3x1080x100xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]
    // CHECK-SAME:   : tensor<2x3x1080x100xf16>, tensor<2x3x1080x100xf16>, tensor<2x3x1080x100xf16> -> tensor<6x3x1080x100xf16>

    // CHECK:       return [[OUTPUT]] : tensor<6x3x1080x100xf16>
}

// -----

func.func @SplitTileModelCase(%arg0: tensor<1x1x1x32xf16>) -> tensor<1x1x65536x32xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [1, 1, 65536, 1]} : tensor<1x1x1x32xf16> -> tensor<1x1x65536x32xf16>
  return %0 : tensor<1x1x65536x32xf16>

    // CHECK-LABEL: @SplitTileModelCase
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x1x32xf16>) -> tensor<1x1x65536x32xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 21846, 1]}
    // CHECK-SAME:   : tensor<1x1x1x32xf16> -> tensor<1x1x21846x32xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 21845, 1]}
    // CHECK-SAME:   : tensor<1x1x1x32xf16> -> tensor<1x1x21845x32xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 21845, 1]}
    // CHECK-SAME:   : tensor<1x1x1x32xf16> -> tensor<1x1x21845x32xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 21846, 0], [0, 0, 43691, 0]
    // CHECK-SAME:   : tensor<1x1x21846x32xf16>, tensor<1x1x21845x32xf16>, tensor<1x1x21845x32xf16> -> tensor<1x1x65536x32xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x65536x32xf16>
}

// -----

func.func @SplitTileNotOnRepeatsDim(%arg0: tensor<1x30x2560x1xf16>) -> tensor<1x30x2560x14xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [1, 1, 1, 14]} : tensor<1x30x2560x1xf16> -> tensor<1x30x2560x14xf16>
  return %0 : tensor<1x30x2560x14xf16>

    // CHECK-LABEL: @SplitTileNotOnRepeatsDim
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x30x2560x1xf16>) -> tensor<1x30x2560x14xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 15, 2560, 1] : tensor<1x30x2560x1xf16> to tensor<1x15x2560x1xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT_TILE0]]) {repeats_values = [1, 1, 1, 14]} : tensor<1x15x2560x1xf16> -> tensor<1x15x2560x14xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 15, 0, 0] [1, 15, 2560, 1] : tensor<1x30x2560x1xf16> to tensor<1x15x2560x1xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT_TILE1]]) {repeats_values = [1, 1, 1, 14]} : tensor<1x15x2560x1xf16> -> tensor<1x15x2560x14xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]]
    // CHECK-SAME:  [0, 0, 0, 0], [0, 15, 0, 0]
    // CHECK-SAME:   : tensor<1x15x2560x14xf16>, tensor<1x15x2560x14xf16> -> tensor<1x30x2560x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x30x2560x14xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SplitTileOnDimH
func.func @SplitTileOnDimH(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x512x512xf16, {order = #NHWC}> {
    %0 = VPU.Tile(%arg0) {repeats_values = [1, 1, 512, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>
    return %0 : tensor<1x32x512x512xf16, {order = #NHWC}>

    // CHECK:         [[TILE_0:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_1:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_2:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_3:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_4:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_5:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_6:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_7:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 57, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x57x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_8:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 56, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x56x512xf16, {order = #NHWC}>
    // CHECK:         [[RET:%.*]] = VPU.Concat([[TILE_0]], [[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]], [[TILE_5]], [[TILE_6]], [[TILE_7]], [[TILE_8]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 57, 0], [0, 0, 114, 0], [0, 0, 171, 0], [0, 0, 228, 0], [0, 0, 285, 0], [0, 0, 342, 0], [0, 0, 399, 0], [0, 0, 456, 0]
    // CHECK:         return [[RET]] : tensor<1x32x512x512xf16, {order = #NHWC}>
}

// -----

func.func @SplitNormalizeL2(%arg0: tensor<3x3x2304x64xf16>) -> tensor<3x3x2304x64xf16> {
  %0 = VPU.NormalizeL2(%arg0) {axes_value = [3], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.eps_mode<MAX>} : tensor<3x3x2304x64xf16> -> tensor<3x3x2304x64xf16>
  return %0 : tensor<3x3x2304x64xf16>
    // CHECK-LABEL:   @SplitNormalizeL2
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<3x3x2304x64xf16>) -> tensor<3x3x2304x64xf16> {

    // CHECK:         [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [3, 3, 768, 64]
    // CHECK-SAME:      : tensor<3x3x2304x64xf16> to tensor<3x3x768x64xf16>
    // CHECK:         [[OUTPUT_TILE0:%.+]] = VPU.NormalizeL2([[INPUT_TILE0]]) {
    // CHECK-SAME:      axes_value = [3], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.eps_mode<MAX>}
    // CHECK-SAME:      : tensor<3x3x768x64xf16> -> tensor<3x3x768x64xf16>

    // CHECK:         [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 768, 0] [3, 3, 768, 64]
    // CHECK-SAME:      : tensor<3x3x2304x64xf16> to tensor<3x3x768x64xf16>
    // CHECK:         [[OUTPUT_TILE1:%.+]] = VPU.NormalizeL2([[INPUT_TILE1]]) {
    // CHECK-SAME:      axes_value = [3], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.eps_mode<MAX>}
    // CHECK-SAME:      : tensor<3x3x768x64xf16> -> tensor<3x3x768x64xf16>

    // CHECK:         [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 1536, 0] [3, 3, 768, 64]
    // CHECK-SAME:      : tensor<3x3x2304x64xf16> to tensor<3x3x768x64xf16>
    // CHECK:         [[OUTPUT_TILE2:%.+]] = VPU.NormalizeL2([[INPUT_TILE2]]) {
    // CHECK-SAME:      axes_value = [3], eps = 9.9999997473787516E-5 : f64, eps_mode = #IE.eps_mode<MAX>}
    // CHECK-SAME:      : tensor<3x3x768x64xf16> -> tensor<3x3x768x64xf16>

    // CHECK:         [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 768, 0], [0, 0, 1536, 0]
    // CHECK-SAME:      : tensor<3x3x768x64xf16>, tensor<3x3x768x64xf16>, tensor<3x3x768x64xf16> -> tensor<3x3x2304x64xf16>

    // CHECK:         return [[OUTPUT]] : tensor<3x3x2304x64xf16>
}

// -----

func.func @SplitNormalizeL2SpecifiedAxis(%arg0: tensor<1x512x37x37xf16>) -> tensor<1x512x37x37xf16> {
  %0 = VPU.NormalizeL2(%arg0) {axes_value = [1], eps = 3.0708054547214041E-41 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x512x37x37xf16> -> tensor<1x512x37x37xf16>
  return %0 : tensor<1x512x37x37xf16>

    // CHECK-LABEL:   @SplitNormalizeL2SpecifiedAxis
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x512x37x37xf16>) -> tensor<1x512x37x37xf16> {

    // CHECK:         [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 19, 37]
    // CHECK-SAME:      : tensor<1x512x37x37xf16> to tensor<1x512x19x37xf16>
    // CHECK:         [[OUTPUT_TILE0:%.+]] = VPU.NormalizeL2([[INPUT_TILE0]]) {
    // CHECK-SAME:      axes_value = [1], eps = 3.0708054547214041E-41 : f64, eps_mode = #IE.eps_mode<ADD>}
    // CHECK-SAME:      : tensor<1x512x19x37xf16> -> tensor<1x512x19x37xf16>

    // CHECK:         [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 19, 0] [1, 512, 18, 37]
    // CHECK-SAME:      : tensor<1x512x37x37xf16> to tensor<1x512x18x37xf16>
    // CHECK:         [[OUTPUT_TILE1:%.+]] = VPU.NormalizeL2([[INPUT_TILE1]]) {
    // CHECK-SAME:      axes_value = [1], eps = 3.0708054547214041E-41 : f64, eps_mode = #IE.eps_mode<ADD>}
    // CHECK-SAME:      : tensor<1x512x18x37xf16> -> tensor<1x512x18x37xf16>

    // CHECK:         [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 19, 0]
    // CHECK-SAME:      : tensor<1x512x19x37xf16>, tensor<1x512x18x37xf16> -> tensor<1x512x37x37xf16>

    // CHECK:         return [[OUTPUT]] : tensor<1x512x37x37xf16>
}

// -----

func.func @DontSplitNormalizeL2AllAxes(%arg0: tensor<1x24x128x128xf16>) -> tensor<1x24x128x128xf16> {
  %0 = VPU.NormalizeL2(%arg0) {axes_value = [0, 1, 2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>} : tensor<1x24x128x128xf16> -> tensor<1x24x128x128xf16>
  return %0 : tensor<1x24x128x128xf16>

  // CHECK-LABEL:   @DontSplitNormalizeL2AllAxes
  // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x24x128x128xf16>) -> tensor<1x24x128x128xf16> {

  // CHECK:         [[OUTPUT:%.+]] = VPU.NormalizeL2([[INPUT]]) {
  // CHECK-SAME:      axes_value = [0, 1, 2, 3], eps = 9.9999999392252903E-9 : f64, eps_mode = #IE.eps_mode<ADD>}
  // CHECK-SAME:      : tensor<1x24x128x128xf16> -> tensor<1x24x128x128xf16>

  // CHECK:         return [[OUTPUT]] : tensor<1x24x128x128xf16>
}

// -----

func.func @SplitTopK(%arg0: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    %aux = const.Declare tensor<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x512x512xf16>, tensor<1x1x512x512xsi32>
    %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    return %0, %target_shape : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    // CHECK: [[AUX:%.+]] = const.Declare tensor<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    // CHECK: [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE0:%.+]], [[TARGET_TILE0:%.+]] = VPU.TopK([[INPUT_TILE0]], [[AUX]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE1:%.+]], [[TARGET_TILE1:%.+]] = VPU.TopK([[INPUT_TILE1]], [[AUX]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 342, 0] [1, 5, 170, 512] : tensor<1x5x512x512xf16> to tensor<1x5x170x512xf16>
    // CHECK: [[OUTPUT_TILE2:%.+]], [[TARGET_TILE2:%.+]] = VPU.TopK([[INPUT_TILE2]], [[AUX]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x170x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x170x512xf16>, tensor<1x1x170x512xsi32>
    // CHECK: [[OUTPUT_VALUE:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]]) {static_offsets =
    // CHECK: [[TARGET_SHAPE:%.+]] = VPU.Concat([[TARGET_TILE0]], [[TARGET_TILE1]], [[TARGET_TILE2]]) {static_offsets =
    // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>
}

// -----

// CHECK-LABEL: func.func @SplitStridedSliceOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x640xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 1280], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x640xf16>
    return %0 : tensor<1x8x80x640xf16>
  }

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.StridedSlice([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x320xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.StridedSlice([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x320xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 320]
// CHECK-SAME:  : tensor<1x8x80x320xf16>, tensor<1x8x80x320xf16> -> tensor<1x8x80x640xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x640xf16>


// -----

func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalNotEltwiseSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalNot([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalNot([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOCInputOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x32x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOCInputOversize(%arg0: tensor<1x512x32x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1],
      ppe = #VPU.PPEStub<>
  } : tensor<1x512x32x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x32x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

  // CHECK-DAG:    [[CONST:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:    [[CONST_0:%.+]] = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[128, 0, 0, 0], [128, 512, 3, 3]>, #const.Reorder<#NHWC>]
  // CHECK-DAG:    [[CONST_1:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:    [[CONST_2:%.+]] = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [128, 512, 3, 3]>, #const.Reorder<#NHWC>]
  // CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 8, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[CONST_2]], [[CONST_1]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 6, 0] [1, 512, 9, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[CONST_2]], [[CONST_1]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 13, 0] [1, 512, 8, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[CONST_2]], [[CONST_1]])
  // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:   rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 19, 0] [1, 512, 8, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[CONST_2]], [[CONST_1]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 25, 0] [1, 512, 7, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x7x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_4:%.+]] = VPU.NCE.Convolution([[SLICE4]], [[CONST_2]], [[CONST_1]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 8, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_5:%.+]] = VPU.NCE.Convolution([[SLICE5]], [[CONST_0]], [[CONST]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 6, 0] [1, 512, 9, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_6:%.+]] = VPU.NCE.Convolution([[SLICE6]], [[CONST_0]], [[CONST]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE7:%.+]] = VPU.Slice [[INPUT]] [0, 0, 13, 0] [1, 512, 8, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_7:%.+]] = VPU.NCE.Convolution([[SLICE7]], [[CONST_0]], [[CONST]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE8:%.+]] = VPU.Slice [[INPUT]] [0, 0, 19, 0] [1, 512, 8, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_8:%.+]] = VPU.NCE.Convolution([[SLICE8]], [[CONST_0]], [[CONST]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[SLICE9:%.+]] = VPU.Slice [[INPUT]] [0, 0, 25, 0] [1, 512, 7, 64]
  // CHECK-SAME:  : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x7x64xf16, {order = #NHWC}>
  // CHECK:    [[CONV_9:%.+]] = VPU.NCE.Convolution([[SLICE9]], [[CONST_0]], [[CONST]])
  // CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
  // CHECK-SAME:  rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:  -> tensor<1x128x6x64xf16, {order = #NHWC}>
  // CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]], [[CONV_4]], [[CONV_5]], [[CONV_6]], [[CONV_7]], [[CONV_8]], [[CONV_9]])
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 7, 0], [0, 0, 14, 0], [0, 0, 20, 0], [0, 0, 26, 0], [0, 128, 0, 0], [0, 128, 7, 0], [0, 128, 14, 0], [0, 128, 20, 0], [0, 128, 26, 0]
  // CHECK-SAME:  : tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}>, tensor<1x128x6x64xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:    return [[CONCAT]] : tensor<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOHFilterOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOHFilterOversize(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1],
      ppe = #VPU.PPEStub<>
  } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

  // CHECK-DAG:       %cst = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:       %cst_0 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[128, 0, 0, 0], [128, 512, 3, 3]>, #const.Reorder<#NHWC>]
  // CHECK-DAG:       %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:       %cst_2 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [128, 512, 3, 3]>, #const.Reorder<#NHWC>]
  // CHECK:       %0 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %2 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %3 = VPU.NCE.Convolution(%2, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %4 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %5 = VPU.NCE.Convolution(%4, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %6 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %8 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %9 = VPU.NCE.Convolution(%8, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %10 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %11 = VPU.NCE.Convolution(%10, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %12 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %13 = VPU.NCE.Convolution(%12, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %14 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %15 = VPU.NCE.Convolution(%14, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %16 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %17 = VPU.NCE.Convolution(%16, %cst_2, %cst_1) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %18 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %19 = VPU.NCE.Convolution(%18, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %20 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %21 = VPU.NCE.Convolution(%20, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %22 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %23 = VPU.NCE.Convolution(%22, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %24 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %25 = VPU.NCE.Convolution(%24, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %26 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %27 = VPU.NCE.Convolution(%26, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %28 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %29 = VPU.NCE.Convolution(%28, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %30 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %31 = VPU.NCE.Convolution(%30, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %32 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %33 = VPU.NCE.Convolution(%32, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %34 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %35 = VPU.NCE.Convolution(%34, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]}
  // CHECK-SAME:       -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %36 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31, %33, %35)
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 15, 0], [0, 0, 22, 0], [0, 0, 29, 0], [0, 0, 36, 0], [0, 0, 43, 0], [0, 0, 50, 0], [0, 0, 57, 0], [0, 128, 0, 0], [0, 128, 8, 0], [0, 128, 15, 0], [0, 128, 22, 0], [0, 128, 29, 0], [0, 128, 36, 0], [0, 128, 43, 0], [0, 128, 50, 0], [0, 128, 57, 0]
  // CHECK-SAME:  : tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:       return %36 : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOCKeepInputShape
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14x!qElemType, {order = #NHWC}>
func.func  @SplitQuantNCEConvOverOCKeepInputShape(%arg0: tensor<1x1024x14x14x!qElemType, {order = #NHWC}>) -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<1024x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<1024x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [1024, 1024, 1, 1],
        strides = [2, 2],
        ppe = #VPU.PPEStub<>
    } : tensor<1x1024x14x14x!qElemType, {order = #NHWC}>, tensor<1024x1024x1x1x!qElemType2, {order = #NHWC}>, tensor<1024x1x1x4xsi32, {order = #NCHW}> -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>, [#const.SubView<[512, 0, 0, 0], [512, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<512x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>, [#const.SubView<[512, 0, 0, 0], [512, 1024, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<10> : tensor<1024x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [512, 1, 1, 4]>]
    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<512x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [512, 1024, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [512, 1024, 1, 1],
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          -> tensor<1x512x7x7x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [512, 1024, 1, 1],
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          -> tensor<1x512x7x7x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:          -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>
}

// -----

func.func @SplitSquaredDiffEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.SquaredDiff(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitSquaredDiffEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SquaredDiff([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SquaredDiff([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

// CHECK-LABEL: @ReLUSplitOverC
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
func.func @ReLUSplitOverC(%arg0: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16> -> tensor<1x80x80x80xf16>
  return %0 : tensor<1x80x80x80xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 48, 80, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16> to tensor<1x48x80x80xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x48x80x80xf16> -> tensor<1x48x80x80xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 48, 0, 0] [1, 32, 80, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16> to tensor<1x32x80x80xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x32x80x80xf16> -> tensor<1x32x80x80xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 48, 0, 0]
// CHECK-SAME:  : tensor<1x48x80x80xf16>, tensor<1x32x80x80xf16> -> tensor<1x80x80x80xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReLUSplitOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}> {
func.func @ReLUSplitOverH(%arg0: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}> {
  %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>
  return %0 : tensor<1x80x80x80xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 80, 40, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16, {order = #NHWC}> to tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 80, 40, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16, {order = #NHWC}> to tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}>, tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL: func.func @NoBiasConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>
func.func @NoBiasConv(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>)
          -> tensor<1x256x62x62xf16> {
    %1 = VPU.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16> -> tensor<1x256x62x62xf16>
    return %1 : tensor<1x256x62x62xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [0, 0]
    // CHECK-SAME:          pads_end = [0, 0]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x62x62xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [0, 0]
    // CHECK-SAME:          pads_end = [0, 0]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x62x62xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x62x62xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x62x62xf16>
}

// -----

// CHECK-LABEL: @DFT
// CHECK-SAME:   (%arg0: tensor<1x100x64x64x2xf16>) -> tensor<1x100x64x64x2xf16>
func.func @DFT(%arg0: tensor<1x100x64x64x2xf16>) -> tensor<1x100x64x64x2xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.DFT(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x100x64x64x2xf16>, tensor<16384xf16> -> tensor<1x100x64x64x2xf16>
    return %0 : tensor<1x100x64x64x2xf16>
    // CHECK: [[CST:%.+]] = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
    // CHECK: [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 50, 64, 64, 2] : tensor<1x100x64x64x2xf16> to tensor<1x50x64x64x2xf16>
    // CHECK: [[FFT0:%.+]] = VPU.DFT([[SLICE0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64x2xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK: [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 50, 0, 0, 0] [1, 50, 64, 64, 2] : tensor<1x100x64x64x2xf16> to tensor<1x50x64x64x2xf16>
    // CHECK: [[FFT1:%.+]] = VPU.DFT([[SLICE1]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64x2xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[FFT0]], [[FFT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0, 0], [0, 50, 0, 0, 0]]} : tensor<1x50x64x64x2xf16>, tensor<1x50x64x64x2xf16> -> tensor<1x100x64x64x2xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x100x64x64x2xf16>
}

// CHECK-LABEL: @IDFT
// CHECK-SAME:   (%arg0: tensor<1x100x64x64x2xf16>) -> tensor<1x100x64x64x2xf16>
func.func @IDFT(%arg0: tensor<1x100x64x64x2xf16>) -> tensor<1x100x64x64x2xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.IDFT(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x100x64x64x2xf16>, tensor<16384xf16> -> tensor<1x100x64x64x2xf16>
    return %0 : tensor<1x100x64x64x2xf16>
    // CHECK: [[CST:%.+]] = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
    // CHECK: [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 50, 64, 64, 2] : tensor<1x100x64x64x2xf16> to tensor<1x50x64x64x2xf16>
    // CHECK: [[FFT0:%.+]] = VPU.IDFT([[SLICE0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64x2xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK: [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 50, 0, 0, 0] [1, 50, 64, 64, 2] : tensor<1x100x64x64x2xf16> to tensor<1x50x64x64x2xf16>
    // CHECK: [[FFT1:%.+]] = VPU.IDFT([[SLICE1]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64x2xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[FFT0]], [[FFT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0, 0], [0, 50, 0, 0, 0]]} : tensor<1x50x64x64x2xf16>, tensor<1x50x64x64x2xf16> -> tensor<1x100x64x64x2xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x100x64x64x2xf16>
}

// -----

// CHECK-LABEL: @RDFTUncut
// CHECK-SAME:  (%arg0: tensor<1x100x64x64xf16>) -> tensor<1x100x64x33x2xf16>
func.func @RDFTUncut(%arg0: tensor<1x100x64x64xf16>) -> tensor<1x100x64x33x2xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %1 = VPU.RDFTUncut(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x100x64x64xf16>, tensor<16384xf16> -> tensor<1x100x64x64x2xf16>
    %2 = VPU.Slice %1 [0, 0, 0, 0, 0] [1, 100, 64, 33, 2] : tensor<1x100x64x64x2xf16> to tensor<1x100x64x33x2xf16>
    return %2 : tensor<1x100x64x33x2xf16>
    // CHECK: [[CST:%.+]] = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
    // CHECK: [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 50, 64, 64] : tensor<1x100x64x64xf16> to tensor<1x50x64x64xf16>
    // CHECK:  [[FFT0:%.+]] = VPU.RDFTUncut([[SLICE0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK:  [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 50, 0, 0] [1, 50, 64, 64] : tensor<1x100x64x64xf16> to tensor<1x50x64x64xf16>
    // CHECK:  [[FFT1:%.+]] = VPU.RDFTUncut([[SLICE1]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x50x64x64xf16>, tensor<16384xf16> -> tensor<1x50x64x64x2xf16>
    // CHECK:  [[CONCAT:%.+]] = VPU.Concat([[FFT0]], [[FFT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0, 0], [0, 50, 0, 0, 0]]} : tensor<1x50x64x64x2xf16>, tensor<1x50x64x64x2xf16> -> tensor<1x100x64x64x2xf16>
    // CHECK:  [[OUTPUT:%.+]] = VPU.Slice [[CONCAT]] [0, 0, 0, 0, 0] [1, 100, 64, 33, 2] : tensor<1x100x64x64x2xf16> to tensor<1x100x64x33x2xf16>
    // CHECK:  return [[OUTPUT]] : tensor<1x100x64x33x2xf16>
}

// -----

// CHECK-LABEL: @IRDFTLastAxis
// CHECK-SAME: (%arg0: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16>
func.func @IRDFTLastAxis(%arg0: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.IRDFTLastAxis(%arg0, %cst) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x120x64x33x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64xf16>
    return %0 : tensor<1x120x64x64xf16>
    // CHECK: [[CST:%.+]] = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
    // CHECK: [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 60, 64, 33, 2] : tensor<1x120x64x33x2xf16> to tensor<1x60x64x33x2xf16>
    // CHECK: [[FFT0:%.+]] = VPU.IRDFTLastAxis([[SLICE0]], [[CST]]) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x60x64x33x2xf16>, tensor<16384xf16> -> tensor<1x60x64x64xf16>
    // CHECK: [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 60, 0, 0, 0] [1, 60, 64, 33, 2] : tensor<1x120x64x33x2xf16> to tensor<1x60x64x33x2xf16>
    // CHECK: [[FFT1:%.+]] = VPU.IRDFTLastAxis([[SLICE1]], [[CST]]) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x60x64x33x2xf16>, tensor<16384xf16> -> tensor<1x60x64x64xf16>
    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[FFT0]], [[FFT1]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 60, 0, 0]]} : tensor<1x60x64x64xf16>, tensor<1x60x64x64xf16> -> tensor<1x120x64x64xf16>
    // CHECK: return [[OUTPUT]] : tensor<1x120x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @MVNTileOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2688x512x1xf16>

func.func @MVNTileOverC(%arg0: tensor<1x2688x512x1xf16>) -> tensor<1x2688x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
    return %0 : tensor<1x2688x512x1xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 896, 512, 1] : tensor<1x2688x512x1xf16> to tensor<1x896x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x896x512x1xf16> -> tensor<1x896x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 896, 0, 0] [1, 896, 512, 1] : tensor<1x2688x512x1xf16> to tensor<1x896x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x896x512x1xf16> -> tensor<1x896x512x1xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 1792, 0, 0] [1, 896, 512, 1] : tensor<1x2688x512x1xf16> to tensor<1x896x512x1xf16>
// CHECK:    [[MVN2:%.+]] = VPU.MVN([[SLICE2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x896x512x1xf16> -> tensor<1x896x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]], [[MVN2]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 896, 0, 0], [0, 1792, 0, 0]]} : tensor<1x896x512x1xf16>, tensor<1x896x512x1xf16>, tensor<1x896x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<1x2688x512x1xf16>
}

// -----

// CHECK-LABEL: func.func @DistributedMVNTileOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2688x512x1xf16>

func.func @DistributedMVNTileOverC(%arg0: tensor<1x2688x512x1xf16>) -> tensor<1x2688x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64,  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
    return %0 : tensor<1x2688x512x1xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1344, 512, 1] : tensor<1x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 1344, 0, 0] [1, 1344, 512, 1] : tensor<1x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 1344, 0, 0]]} : tensor<1x1344x512x1xf16>, tensor<1x1344x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<1x2688x512x1xf16>
}

// -----

// CHECK-LABEL: func.func @MVNTileOverNC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x1344x512x1xf16>

func.func @MVNTileOverNC(%arg0: tensor<2x1344x512x1xf16>) -> tensor<2x1344x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<2x1344x512x1xf16> -> tensor<2x1344x512x1xf16>
    return %0 : tensor<2x1344x512x1xf16>
// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 672, 512, 1] : tensor<2x1344x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 672, 0, 0] [1, 672, 512, 1] : tensor<2x1344x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 672, 512, 1] : tensor<2x1344x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN2:%.+]] = VPU.MVN([[SLICE2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [1, 672, 0, 0] [1, 672, 512, 1] : tensor<2x1344x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN3:%.+]] = VPU.MVN([[SLICE3]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]], [[MVN2]], [[MVN3]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 672, 0, 0], [1, 0, 0, 0], [1, 672, 0, 0]]} : tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16> -> tensor<2x1344x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<2x1344x512x1xf16>
}

// -----

// CHECK-LABEL: func.func @DistributedMVNTileOverNC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x2688x512x1xf16>

func.func @DistributedMVNTileOverNC(%arg0: tensor<2x2688x512x1xf16>) -> tensor<2x2688x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<2x2688x512x1xf16> -> tensor<2x2688x512x1xf16>
    return %0 : tensor<2x2688x512x1xf16>
// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1344, 512, 1] : tensor<2x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 1344, 0, 0] [1, 1344, 512, 1] : tensor<2x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 1344, 512, 1] : tensor<2x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN2:%.+]] = VPU.MVN([[SLICE2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [1, 1344, 0, 0] [1, 1344, 512, 1] : tensor<2x2688x512x1xf16> to tensor<1x1344x512x1xf16>
// CHECK:    [[MVN3:%.+]] = VPU.MVN([[SLICE3]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x1344x512x1xf16> -> tensor<1x1344x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]], [[MVN2]], [[MVN3]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 1344, 0, 0], [1, 0, 0, 0], [1, 1344, 0, 0]]} : tensor<1x1344x512x1xf16>, tensor<1x1344x512x1xf16>, tensor<1x1344x512x1xf16>, tensor<1x1344x512x1xf16> -> tensor<2x2688x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<2x2688x512x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1NormalizeSplit
// CHECK-SAME:        [[INPUT0:%arg[0-9]]]: tensor<1x1x1x520001xf16>
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x1x2xf16, {order = #NHWC}>
func.func @MVN1NormalizeSplit(%arg0: tensor<1x1x1x520001xf16>, %arg1: tensor<1x1x1x2xf16, {order = #NHWC}>) -> tensor<1x1x1x520001xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>
    return %0 : tensor<1x1x1x520001xf16>

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 1, 1, 173334] : tensor<1x1x1x520001xf16> to tensor<1x1x1x173334xf16>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1Normalize([[INPUT_TILE_1]], [[INPUT1]])
    // CHECK-SAME:     :  tensor<1x1x1x173334xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x173334xf16>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 173334] [1, 1, 1, 173334] : tensor<1x1x1x520001xf16> to tensor<1x1x1x173334xf16>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1Normalize([[INPUT_TILE_2]], [[INPUT1]])
    // CHECK-SAME:     :  tensor<1x1x1x173334xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x173334xf16>

    // CHECK:       [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 346668] [1, 1, 1, 173333] : tensor<1x1x1x520001xf16> to tensor<1x1x1x173333xf16>
    // CHECK:       [[OUTPUT_TILE_3:%.+]] = VPU.MVN1Normalize([[INPUT_TILE_3]], [[INPUT1]])
    // CHECK-SAME:     :  tensor<1x1x1x173333xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x173333xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]], [[OUTPUT_TILE_3]])
    // CHECK:       return [[CONCAT]] :  tensor<1x1x1x520001xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @NCEPermute
func.func @NCEPermute(%arg0: tensor<1x3x32x8000xf16>) -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 16, 8000]
    // CHECK-SAME:      : tensor<1x3x32x8000xf16> to tensor<1x3x16x8000xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Permute([[INPUT_TILE0]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 4 : i64
    // CHECK-SAME:      -> tensor<1x4x16x8000x!qElemType, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 16, 0] [1, 3, 16, 8000]
    // CHECK-SAME:      : tensor<1x3x32x8000xf16> to tensor<1x3x16x8000xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Permute([[INPUT_TILE1]])
    // CHECK-SAME:          dstElemType = !qElemType,
    // CHECK-SAME:          dstOrder = #NHWC,
    // CHECK-SAME:          expandedChannels = 4 : i64
    // CHECK-SAME:      -> tensor<1x4x16x8000x!qElemType, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 16, 0]
    // CHECK-SAME:      -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x4x32x8000x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: func.func @SplitNCEPermuteOverWidth
func.func @SplitNCEPermuteOverWidth(%arg0: tensor<1x2048x1x496xf16>) -> tensor<1x2048x1x496xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 2048 : i64,
        ppe = #VPU.PPEStub<>
        } -> tensor<1x2048x1x496xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x1x496xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_0:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 2048, 1, 176] :
    // CHECK-SAME:      tensor<1x2048x1x496xf16> to tensor<1x2048x1x176xf16>
    // CHECK:       [[NCE_PERM_0:%.*]] = VPU.NCE.Permute([[SLICE_0]])
    // CHECK-SAME:      -> tensor<1x2048x1x176xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_1:%.*]] = VPU.Slice %arg0 [0, 0, 0, 176] [1, 2048, 1, 176] :
    // CHECK-SAME:      tensor<1x2048x1x496xf16> to tensor<1x2048x1x176xf16>
    // CHECK:       [[NCE_PERM_1:%.*]] = VPU.NCE.Permute([[SLICE_1]])
    // CHECK-SAME:      -> tensor<1x2048x1x176xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_2:%.*]] = VPU.Slice %arg0 [0, 0, 0, 352] [1, 2048, 1, 144] :
    // CHECK-SAME:      tensor<1x2048x1x496xf16> to tensor<1x2048x1x144xf16>
    // CHECK:       [[NCE_PERM_2:%.*]] = VPU.NCE.Permute([[SLICE_2]])
    // CHECK-SAME:      -> tensor<1x2048x1x144xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[NCE_PERM_0]], [[NCE_PERM_1]], [[NCE_PERM_2]])
    // CHECK-SAME:      {static_offsets = [
    // CHECK-SAME:        [0, 0, 0, 0], [0, 0, 0, 176], [0, 0, 0, 352]]} :
    // CHECK-SAME:      tensor<1x2048x1x176xf16, {order = #NHWC}>, tensor<1x2048x1x176xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x2048x1x144xf16
    // CHECK-SAME:      -> tensor<1x2048x1x496xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT]] : tensor<1x2048x1x496xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @TilePadOverWidth
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x227x240xf16>

func.func @TilePadOverWidth(%arg0: tensor<1x3x227x240xf16>) -> tensor<1x16x227x240xf16> {
    %pad = VPU.Pad(%arg0)
          {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
          pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
          : tensor<1x3x227x240xf16> -> tensor<1x16x227x240xf16>

    return %pad : tensor<1x16x227x240xf16>

    // Tile 0

    // CHECK:       [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 227, 120]
    // CHECK-SAME:      : tensor<1x3x227x240xf16> to tensor<1x3x227x120xf16>

    // CHECK:       [[PAD0:%.*]] = VPU.Pad([[SLICE0]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x120xf16> -> tensor<1x16x227x120xf16>

    // Tile 1

    // CHECK:       [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 120] [1, 3, 227, 120]
    // CHECK-SAME:      : tensor<1x3x227x240xf16> to tensor<1x3x227x120xf16>

    // CHECK:       [[PAD1:%.*]] = VPU.Pad([[SLICE1]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x120xf16> -> tensor<1x16x227x120xf16>

    // Concat

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[PAD0]], [[PAD1]]) {static_offsets =
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 120]
    // CHECK-SAME:      : tensor<1x16x227x120xf16>, tensor<1x16x227x120xf16> -> tensor<1x16x227x240xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x16x227x240xf16>
}

// -----

// CHECK-LABEL: func.func @TilePadOverPaddedWidth
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x3x227x580xf16>

func.func @TilePadOverPaddedWidth(%arg0: tensor<1x3x227x580xf16>) -> tensor<1x16x227x600xf16> {
    %pad = VPU.Pad(%arg0)
          {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
          pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 13, 0, 10]}
          : tensor<1x3x227x580xf16> -> tensor<1x16x227x600xf16>

    return %pad : tensor<1x16x227x600xf16>

    // Tile 0

    // CHECK:       [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 227, 190]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x190xf16>

    // CHECK:       [[PAD0:%.*]] = VPU.Pad([[SLICE0]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x190xf16> -> tensor<1x16x227x200xf16>

    // Tile 1

    // CHECK:       [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 190] [1, 3, 227, 200]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x200xf16>

    // CHECK:       [[PAD1:%.*]] = VPU.Pad([[SLICE1]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x200xf16> -> tensor<1x16x227x200xf16>

    // Tile 2

    // CHECK:       [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 390] [1, 3, 227, 190]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x190xf16>

    // CHECK:       [[PAD2:%.*]] = VPU.Pad([[SLICE2]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 10]}
    // CHECK-SAME:      : tensor<1x3x227x190xf16> -> tensor<1x16x227x200xf16>

    // Concat

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[PAD0]], [[PAD1]], [[PAD2]]) {static_offsets =
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 200], [0, 0, 0, 400]
    // CHECK-SAME:       : tensor<1x16x227x200xf16>, tensor<1x16x227x200xf16>, tensor<1x16x227x200xf16> -> tensor<1x16x227x600xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x16x227x600xf16>
}
