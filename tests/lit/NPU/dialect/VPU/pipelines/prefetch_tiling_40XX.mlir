//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --tiling="enable-prefetch=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: func.func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x24x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @SplitSwConvOverOC(
        %input: tensor<1x24x64x64xf16>,
        %filter: tensor<256x24x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x24x64x64xf16>, tensor<256x24x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 24, 3, 3]
    // CHECK-SAME:      : tensor<256x24x3x3xf16> to tensor<128x24x3x3xf16>

    // CHECK:       [[BIAS_TILE0:%.+]] = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 24, 3, 3]
    // CHECK-SAME:      : tensor<256x24x3x3xf16> to tensor<128x24x3x3xf16>

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
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x168xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x244x168xf16>)
            -> tensor<1x16x244x168xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x244x168xf16> -> tensor<1x16x244x168xf16>
    return %1 : tensor<1x16x244x168xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 123, 168
    // CHECK-SAME:       : tensor<1x16x244x168xf16> to tensor<1x16x123x168xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x168xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 121, 0] [1, 16, 123, 168
    // CHECK-SAME:      : tensor<1x16x244x168xf16> to tensor<1x16x123x168xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x168xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 122, 0]
    // CHECK-SAME:      -> tensor<1x16x244x168xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x168xf16>
}

// -----

// CHECK-LABEL: func @SplitSoftMaxOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x384xf16>
func.func @SplitSoftMaxOverW(%arg0: tensor<1x20x256x384xf16>) -> tensor<1x20x256x384xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x384xf16> -> tensor<1x20x256x384xf16>
    return %0 : tensor<1x20x256x384xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftMax([[INPUT_TILE0]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 64] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftMax([[INPUT_TILE1]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 128] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.SoftMax([[INPUT_TILE2]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 192] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.SoftMax([[INPUT_TILE3]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 256] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.SoftMax([[INPUT_TILE4]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 320] [1, 20, 256, 64]
    // CHECK-SAME:      : tensor<1x20x256x384xf16> to tensor<1x20x256x64xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.SoftMax([[INPUT_TILE5]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x64xf16> -> tensor<1x20x256x64xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 64], [0, 0, 0, 128], [0, 0, 0, 192], [0, 0, 0, 256], [0, 0, 0, 320]
    // CHECK-SAME:      -> tensor<1x20x256x384xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x384xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x24x64x64xf16>)
            -> tensor<1x24x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
            tensor<1x24x64x64xf16> -> tensor<1x24x256x256xf16>

    return %0 : tensor<1x24x256x256xf16>
}

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 24, 22, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x22x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x24x22x64xf16>
// CHECK-SAME:      -> tensor<1x24x86x256xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 21, 0] [1, 24, 23, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x23x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x24x23x64xf16>
// CHECK-SAME:      -> tensor<1x24x85x256xf16>
// CHECK:       [[TILE2:%.+]] = VPU.Slice %arg0 [0, 0, 42, 0] [1, 24, 22, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x22x64xf16>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x24x22x64xf16>
// CHECK-SAME:      -> tensor<1x24x85x256xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]]}
// CHECK-SAME:      : tensor<1x24x86x256xf16>, tensor<1x24x85x256xf16>, tensor<1x24x85x256xf16> -> tensor<1x24x256x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>

// -----


// CHECK-LABEL: func.func @InterpSplitOverCDueToUglyScalingFactor
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x21x65x65xf16>
func.func @InterpSplitOverCDueToUglyScalingFactor(%arg0: tensor<1x21x65x65xf16>) -> tensor<1x21x513x513xf16> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
    axes_attr = [2, 3],
    multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
    scales_attr = [7.8923077583312988, 7.8923077583312988],
    sizes_attr = [513, 513]} : tensor<1x21x65x65xf16> -> tensor<1x21x513x513xf16>
  return %0 : tensor<1x21x513x513xf16>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 7, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE2:%.+]] = VPU.Slice %arg0 [0, 0, 14, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE3:%.+]] = VPU.Slice %arg0 [0, 0, 21, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP3:%.+]] = VPU.Interpolate([[TILE3]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE4:%.+]] = VPU.Slice %arg0 [0, 0, 28, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP4:%.+]] = VPU.Interpolate([[TILE4]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE5:%.+]] = VPU.Slice %arg0 [0, 0, 36, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP5:%.+]] = VPU.Interpolate([[TILE5]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE6:%.+]] = VPU.Slice %arg0 [0, 0, 43, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP6:%.+]] = VPU.Interpolate([[TILE6]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE7:%.+]] = VPU.Slice %arg0 [0, 0, 50, 0] [1, 21, 9, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x9x65xf16>
// CHECK:       [[INTERP7:%.+]] = VPU.Interpolate([[TILE7]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x9x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[TILE8:%.+]] = VPU.Slice %arg0 [0, 0, 57, 0] [1, 21, 8, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16> to tensor<1x21x8x65xf16>
// CHECK:       [[INTERP8:%.+]] = VPU.Interpolate([[TILE8]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x21x8x65xf16>
// CHECK-SAME:      -> tensor<1x21x57x513xf16>
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]], [[INTERP3]], [[INTERP4]], [[INTERP5]], [[INTERP6]], [[INTERP7]], [[INTERP8]]) {
// CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 0, 57, 0], [0, 0, 114, 0], [0, 0, 171, 0], [0, 0, 228, 0], [0, 0, 285, 0], [0, 0, 342, 0], [0, 0, 399, 0], [0, 0, 456, 0]]}
// CHECK-SAME:      : tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>, tensor<1x21x57x513xf16>
// CHECK-SAME:      -> tensor<1x21x513x513xf16>
// CHECK:       return [[CONCAT]] : tensor<1x21x513x513xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCDueTo1Size
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @InterpSplitOverCDueTo1Size(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x65x65xf16, {order = #NHWC}> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
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
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
func.func @SplitPReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.PRelu([[INPUT_TILE0]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480]  [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.PRelu([[INPUT_TILE1]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

  }

// -----

// CHECK-LABEL: func.func @SplitLeakyReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LeakyRelu([[INPUT_TILE0]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LeakyRelu([[INPUT_TILE1]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<576x1x1x4xsi32, {order = #NHWC}>
func.func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<576x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<576x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x576x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [576, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}>, tensor<576x1x1x4xsi32, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:    strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS2]] [0, 0, 0, 0] [144, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<144x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE0:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [0, 0, 0, 0] [144, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<144x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:    rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:    strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS2]] [144, 0, 0, 0] [144, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<144x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE1:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [144, 0, 0, 0] [144, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<144x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [144, 144, 3, 3],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[WEIGHTS_TILE2:%.+]] = VPU.Slice [[WEIGHTS2]] [288, 0, 0, 0] [144, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<144x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE2:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [288, 0, 0, 0] [144, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<144x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE2]], [[WEIGHTS_TABLE_TILE2]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>


    // Tile 3

    // CHECK:       [[WEIGHTS_TILE3:%.+]] = VPU.Slice [[WEIGHTS2]] [432, 0, 0, 0] [144, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<144x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE3:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [432, 0, 0, 0] [144, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<144x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE3]], [[WEIGHTS_TABLE_TILE3]])
    // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 144, 0, 0], [0, 288, 0, 0], [0, 432, 0, 0]
    // CHECK-SAME:      -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x48xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x48xf16, {order = #NHWC}>) -> tensor<1x256x64x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x48xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x48xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 48]
    // CHECK-SAME:      : tensor<1x32x64x48xf16, {order = #NHWC}> to tensor<1x32x33x48xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x256x32x48xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 48]
    // CHECK-SAME:      : tensor<1x32x64x48xf16, {order = #NHWC}> to tensor<1x32x33x48xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHEKC-SAME:          strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x48xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x256x64x48xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x48xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

// CHECK-LABEL:   @SplitI4QuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
func.func @SplitI4QuantNCEConvOverOC(%arg0: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<6320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<6320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
    } -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<2096x1x1x4xsi32> = dense<10> : tensor<6320x1x1x4xsi32>, [#const.SubView<[4224, 0, 0, 0], [2096, 1, 1, 4]>]
    // CHECK-DAG:       [[CST_0:%.+]] = const.Declare tensor<2096x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.SubView<[4224, 0, 0, 0], [2096, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[CST_1:%.+]] = const.Declare tensor<2112x1x1x4xsi32> = dense<10> : tensor<6320x1x1x4xsi32>, [#const.SubView<[2112, 0, 0, 0], [2112, 1, 1, 4]>]
    // CHECK-DAG:       [[CST_2:%.+]] = const.Declare tensor<2112x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.SubView<[2112, 0, 0, 0], [2112, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[CST_3:%.+]] = const.Declare tensor<2112x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [2112, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[CST_4:%.+]] = const.Declare tensor<2112x1x1x4xsi32> = dense<10> : tensor<6320x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [2112, 1, 1, 4]>]

    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[CST_3]], [[CST_4]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [2112, 128, 1, 1], strides = [1, 1]} -> tensor<1x2112x256x4xf16, {order = #NHWC}>

    // CHECK:           [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[CST_2]], [[CST_1]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [2112, 128, 1, 1], strides = [1, 1]} -> tensor<1x2112x256x4xf16, {order = #NHWC}>

    // CHECK:           [[CONV_2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[CST_0]], [[CST]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [2096, 128, 1, 1], strides = [1, 1]} -> tensor<1x2096x256x4xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]]) {
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 2112, 0, 0], [0, 4224, 0, 0]]}
    // CHECK-SAME:      : tensor<1x2112x256x4xf16, {order = #NHWC}>, tensor<1x2112x256x4xf16, {order = #NHWC}>, tensor<1x2096x256x4xf16, {order = #NHWC}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:           return [[CONCAT]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x256xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x256xf16, {order = #NHWC}>) -> tensor<1x16x340x256xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x340x256xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 50, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x49x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 48, 0] [1, 16, 51, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x49x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 97, 0] [1, 16, 51, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x49x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 146, 0] [1, 16, 51, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x49x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 195, 0] [1, 16, 50, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x48x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 243, 0] [1, 16, 50, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x48x256xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 291, 0] [1, 16, 49, 256]
    // CHECK-SAME:      : tensor<1x16x340x256xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x49x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE6]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x16x48x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 49, 0], [0, 0, 98, 0], [0, 0, 147, 0], [0, 0, 196, 0], [0, 0, 244, 0], [0, 0, 292, 0]
    // CHECK-SAME:      -> tensor<1x16x340x256xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x256xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

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
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x512xf16, {order = #NHWC}>
func.func @TileWithSOH(
        %arg0: tensor<1x16x210x512xf16, {order = #NHWC}>)
            -> tensor<1x32x210x512xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
        : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x32x210x512xf16, {order = #NHWC}>

    return %0 : tensor<1x32x210x512xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 71, 512]
    // CHECK-SAME:          tensor<1x16x210x512xf16, {order = #NHWC}> to tensor<1x16x71x512xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x70x512xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 69, 0] [1, 16, 72, 512]
    // CHECK-SAME:          tensor<1x16x210x512xf16, {order = #NHWC}> to tensor<1x16x72x512xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x70x512xf16, {order = #NHWC}>

    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 139, 0] [1, 16, 71, 512]
    // CHECK-SAME:          tensor<1x16x210x512xf16, {order = #NHWC}> to tensor<1x16x71x512xf16, {order = #NHWC}>

    // CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x70x512xf16, {order = #NHWC}>


    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]], [[CONV3]])

    // CHECK:       return [[CONCAT]] : tensor<1x32x210x512xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x240x10x10xf16, {order = #NHWC}>

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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [768, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

  // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16,
  // CHECK-SAME:         {order = #NHWC}> = dense<1.000000e+00> : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[WEIGHTS0:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  // CHECK:       [[SLICE:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 18, 30]
  // CHECK-SAME:         tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x18x30xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT0:%.+]] = VPU.NCE.Convolution([[SLICE]], [[WEIGHTS]], [[WEIGHTS0]])
  // CHECK-SAME:         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
  // CHECK-SAME:         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 0 : i64>,
  // CHECK-SAME:         rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
  // CHECK-SAME:         -> tensor<1x768x15x30xf16, {order = #NHWC}>
  // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 12, 0] [1, 32, 18, 30] : tensor<1x32x30x30xf16, {order = #NHWC}> to tensor<1x32x18x30xf16, {order = #NHWC}>
  // CHECK:       [[OUTPUT1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS0]])
  // CHECK-SAME:         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
  // CHECK-SAME:         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 0 : i64, bottom = 3 : i64>,
  // CHECK-SAME:         rawFilterShape = [768, 32, 7, 7], strides = [1, 1]}
  // CHECK-SAME:         -> tensor<1x768x15x30xf16, {order = #NHWC}>
  // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT0]], [[OUTPUT1]])
  // CHECK-LITERAL:       {static_offsets = [[0, 0, 0, 0], [0, 0, 15, 0]]} : tensor<1x768x15x30xf16, {order = #NHWC}>, tensor<1x768x15x30xf16, {order = #NHWC}> -> tensor<1x768x30x30xf16, {order = #NHWC}>
  // CHECK:       return [[CONCAT]] : tensor<1x768x30x30xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>, [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x256x3x3xf16>, [#const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<512x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
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
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x50xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x50xf16, {order = #NHWC}>
func.func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x70x50xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x70x50xf16, {order = #NHWC}>)
            -> tensor<1x64x70x50xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x70x50xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x70x50xf16, {order = #NHWC}>

    return %1 : tensor<1x64x70x50xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK-NOT:   VPU.Slice
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x70x50xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x160x80x60xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_TILE:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_SM_TILE]], [[WEIGHTS_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 60]
    // CHECK-SAME:      : tensor<1x32x80x60xf16, {order = #NHWC}> to tensor<1x32x41x60xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x60xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 60]
    // CHECK-SAME:      : tensor<1x32x80x60xf16, {order = #NHWC}> to tensor<1x32x41x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_2]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x60xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x8640xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x8640xf16, {order = #NHWC}>) -> tensor<1x16x1x8640xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x8640xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x8640xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 2160]
    // CHECK-SAME:      tensor<1x16x7x8640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2160xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2160xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2160] [1, 16, 7, 2160]
    // CHECK-SAME:      tensor<1x16x7x8640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2160xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2160xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4320] [1, 16, 7, 2160]
    // CHECK-SAME:      tensor<1x16x7x8640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2160xf16, {order = #NHWC}>


    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2160xf16, {order = #NHWC}>

    // Tile 3

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 6480] [1, 16, 7, 2160]
    // CHECK-SAME:      tensor<1x16x7x8640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2160xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE3]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2160xf16, {order = #NHWC}>


    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 2160], [0, 0, 0, 4320], [0, 0, 0, 6480]
    // CHECK-SAME:      -> tensor<1x16x1x8640xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8640xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x184320xf16>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x184320xf16>) -> tensor<1x1x1x184320xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x184320xf16> -> tensor<1x1x1x184320xf16>

    return %0 : tensor<1x1x1x184320xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 7, 61440]
    // CHECK-SAME:      : tensor<1x1x7x184320xf16>
    // CHECK-SAME:      to tensor<1x1x7x61440xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1x1x61440xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 61440] [1, 1, 7, 61440]
    // CHECK-SAME:      : tensor<1x1x7x184320xf16>
    // CHECK-SAME:      to tensor<1x1x7x61440xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1x1x61440xf16>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 122880] [1, 1, 7, 61440]
    // CHECK-SAME:      : tensor<1x1x7x184320xf16>
    // CHECK-SAME:      to tensor<1x1x7x61440xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.AvgPool([[INPUT_TILE2]])
    // CHECK-SAME:      -> tensor<1x1x1x61440xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 61440], [0, 0, 0, 122880]
    // CHECK-SAME:      -> tensor<1x1x1x184320xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x184320xf16>
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @LogSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Log(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Log([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Log([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

// CHECK-LABEL: @AbsSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @AbsSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Abs(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Abs([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Abs([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitFloorModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.FloorMod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.FloorMod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
// -----

func.func @SplitModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.Mod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Mod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Mod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
// -----

func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitPowerEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Power([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Power([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitLogicalOrEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalOr([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalOr([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitLogicalXorEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalXor([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalXor([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
  %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
  return %0 : tensor<1x10x256x176xi8>
}

// CHECK-LABEL: @SplitEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Equal([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Equal([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xi8>, tensor<1x10x128x176xi8> -> tensor<1x10x256x176xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>

// -----

func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
  %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
  return %0 : tensor<1x10x256x176xi8>
}

// CHECK-LABEL: @SplitNotEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NotEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NotEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xi8>, tensor<1x10x128x176xi8> -> tensor<1x10x256x176xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>

// -----

func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
  %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
  return %0 : tensor<1x10x256x176xi8>
}

// CHECK-LABEL: @SplitLessEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Less([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Less([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xi8>, tensor<1x10x128x176xi8> -> tensor<1x10x256x176xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>

// -----

func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
  %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
  return %0 : tensor<1x10x256x176xi8>
}

// CHECK-LABEL: @SplitLessEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LessEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LessEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xi8>, tensor<1x10x128x176xi8> -> tensor<1x10x256x176xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>

// -----

func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
  %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
  return %0 : tensor<1x10x256x176xi8>
}

// CHECK-LABEL: @SplitGreaterEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Greater([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Greater([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xi8>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xi8>, tensor<1x10x128x176xi8> -> tensor<1x10x256x176xi8>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>

// -----

func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitGreaterEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.GreaterEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.GreaterEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitErfOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @SplitErfOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Erf(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Erf([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Erf([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

// CHECK-LABEL: @SplitFloorOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @SplitFloorOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Floor(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Floor([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Floor([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @TanSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Tan(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tan([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tan([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

// CHECK-LABEL: @SwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @SwishSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Swish([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Swish([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

// CHECK-LABEL: @HSigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSigmoid([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSigmoid([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Negative(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Negative([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Negative([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Ceiling([[INPUT_TILE0]])
    // CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
    // CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Ceiling([[INPUT_TILE1]])
    // CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
    // CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

}

// -----

func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Sign(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @SplitSignActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sign([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sign([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>, %arg2: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitSelectEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 86, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x86x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 86, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x86x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 0, 0] [1, 10, 86, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x86x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Select([[INPUT_TILE0]], [[INPUT_TILE1]], [[INPUT_TILE2]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x86x176xf16>, tensor<1x10x86x176xf16>, tensor<1x10x86x176xf16> -> tensor<1x10x86x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 86, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 86, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 86, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Select([[INPUT_TILE3]], [[INPUT_TILE4]], [[INPUT_TILE5]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x85x176xf16>, tensor<1x10x85x176xf16>, tensor<1x10x85x176xf16> -> tensor<1x10x85x176xf16>

// CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 171, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 171, 0] [1, 10, 85, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x85x176xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Select([[INPUT_TILE6]], [[INPUT_TILE7]], [[INPUT_TILE8]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x85x176xf16>, tensor<1x10x85x176xf16>, tensor<1x10x85x176xf16> -> tensor<1x10x85x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:  : tensor<1x10x86x176xf16>, tensor<1x10x85x176xf16>, tensor<1x10x85x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

func.func @SplitAddEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitAddEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Round([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Round([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @SplitGeluActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Gelu([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Gelu([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @SplitTopK(%arg0: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>) {
    %output_values, %target_shape = VPU.TopK(%arg0) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x384xf16> -> tensor<1x1x512x384xf16>, tensor<1x1x512x384xsi32>
    %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
    return %0, %target_shape : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>) {
    // CHECK: [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 171, 384] : tensor<1x5x512x384xf16> to tensor<1x5x171x384xf16>
    // CHECK: [[OUTPUT_TILE0:%.+]], [[TARGET_TILE0:%.+]] = VPU.TopK([[INPUT_TILE0]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x384xf16> -> tensor<1x1x171x384xf16>, tensor<1x1x171x384xsi32>
    // CHECK: [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 5, 171, 384] : tensor<1x5x512x384xf16> to tensor<1x5x171x384xf16>
    // CHECK: [[OUTPUT_TILE1:%.+]], [[TARGET_TILE1:%.+]] = VPU.TopK([[INPUT_TILE1]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x384xf16> -> tensor<1x1x171x384xf16>, tensor<1x1x171x384xsi32>
    // CHECK: [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 342, 0] [1, 5, 170, 384] : tensor<1x5x512x384xf16> to tensor<1x5x170x384xf16>
    // CHECK: [[OUTPUT_TILE2:%.+]], [[TARGET_TILE2:%.+]] = VPU.TopK([[INPUT_TILE2]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 0>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x170x384xf16> -> tensor<1x1x170x384xf16>, tensor<1x1x170x384xsi32>
    // CHECK: [[OUTPUT_VALUE:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]]) {static_offsets =
    // CHECK: [[TARGET_SHAPE:%.+]] = VPU.Concat([[TARGET_TILE0]], [[TARGET_TILE1]], [[TARGET_TILE2]]) {static_offsets =
    // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
    // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>
}

// -----

// CHECK-LABEL: func.func @SplitStridedSliceOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x480xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 960], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x480xf16>
    return %0 : tensor<1x8x80x480xf16>
  }

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.StridedSlice([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x240xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.StridedSlice([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x240xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 240]
// CHECK-SAME:  : tensor<1x8x80x240xf16>, tensor<1x8x80x240xf16> -> tensor<1x8x80x480xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x480xf16>

// -----

func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitLogicalNotEltwiseSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalNot([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalNot([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

// CHECK-LABEL: @MVNTileOverC
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x5376x512x1xf16>

func.func @MVNTileOverC(%arg0: tensor<1x5376x512x1xf16>) -> tensor<1x5376x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x5376x512x1xf16> -> tensor<1x5376x512x1xf16>
    return %0 : tensor<1x5376x512x1xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 672, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 1344, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN2:%.+]] = VPU.MVN([[SLICE2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 2016, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN3:%.+]] = VPU.MVN([[SLICE3]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE4:%.+]] = VPU.Slice [[INPUT]] [0, 2688, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN4:%.+]] = VPU.MVN([[SLICE4]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE5:%.+]] = VPU.Slice [[INPUT]] [0, 3360, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN5:%.+]] = VPU.MVN([[SLICE5]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE6:%.+]] = VPU.Slice [[INPUT]] [0, 4032, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN6:%.+]] = VPU.MVN([[SLICE6]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[SLICE7:%.+]] = VPU.Slice [[INPUT]] [0, 4704, 0, 0] [1, 672, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x672x512x1xf16>
// CHECK:    [[MVN7:%.+]] = VPU.MVN([[SLICE7]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x672x512x1xf16> -> tensor<1x672x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]], [[MVN2]], [[MVN3]], [[MVN4]], [[MVN5]], [[MVN6]], [[MVN7]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 672, 0, 0], [0, 1344, 0, 0], [0, 2016, 0, 0], [0, 2688, 0, 0], [0, 3360, 0, 0], [0, 4032, 0, 0], [0, 4704, 0, 0]]} : tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16>, tensor<1x672x512x1xf16> -> tensor<1x5376x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<1x5376x512x1xf16>
}

// -----

// CHECK-LABEL: @DistributedMVNTileOverC
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x5376x512x1xf16>

func.func @DistributedMVNTileOverC(%arg0: tensor<1x5376x512x1xf16>) -> tensor<1x5376x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x5376x512x1xf16> -> tensor<1x5376x512x1xf16>
    return %0 : tensor<1x5376x512x1xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2688, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2688, 0, 0] [1, 2688, 512, 1] : tensor<1x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 2688, 0, 0]]} : tensor<1x2688x512x1xf16>, tensor<1x2688x512x1xf16> -> tensor<1x5376x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<1x5376x512x1xf16>
}

// -----

// CHECK-LABEL: @MVNTileOverNC
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<2x1344x512x1xf16>

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

// CHECK-LABEL: @DistributedMVNTileOverNC
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<2x5376x512x1xf16>

func.func @DistributedMVNTileOverNC(%arg0: tensor<2x5376x512x1xf16>) -> tensor<2x5376x512x1xf16> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<2x5376x512x1xf16> -> tensor<2x5376x512x1xf16>
    return %0 : tensor<2x5376x512x1xf16>
// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2688, 512, 1] : tensor<2x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN0:%.+]] = VPU.MVN([[SLICE0]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2688, 0, 0] [1, 2688, 512, 1] : tensor<2x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN1:%.+]] = VPU.MVN([[SLICE1]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 2688, 512, 1] : tensor<2x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN2:%.+]] = VPU.MVN([[SLICE2]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [1, 2688, 0, 0] [1, 2688, 512, 1] : tensor<2x5376x512x1xf16> to tensor<1x2688x512x1xf16>
// CHECK:    [[MVN3:%.+]] = VPU.MVN([[SLICE3]]) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x2688x512x1xf16> -> tensor<1x2688x512x1xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[MVN0]], [[MVN1]], [[MVN2]], [[MVN3]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 2688, 0, 0], [1, 0, 0, 0], [1, 2688, 0, 0]]} : tensor<1x2688x512x1xf16>, tensor<1x2688x512x1xf16>, tensor<1x2688x512x1xf16>, tensor<1x2688x512x1xf16> -> tensor<2x5376x512x1xf16>
// CHECK:    return [[CONCAT]] : tensor<2x5376x512x1xf16>
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

// CHECK-LABEL: func.func @SplitNCEPermute
func.func @SplitNCEPermute(%arg0: tensor<1x3x32x8000xf16>) -> tensor<1x4x32x8000x!qElemType, {order = #NHWC}> {
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
// CHECK-LABEL: @SplitTileOnDimH
func.func @SplitTileOnDimH(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x512x512xf16, {order = #NHWC}> {
    %0 = VPU.Tile(%arg0) {repeats_values = [1, 1, 512, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x512x512xf16, {order = #NHWC}>
    return %0 : tensor<1x32x512x512xf16, {order = #NHWC}>

    // CHECK:         [[TILE_0:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_1:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_2:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_3:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_4:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_5:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_6:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_7:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 43, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x43x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_8:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 42, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x42x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_9:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 42, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x42x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_10:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 42, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x42x512xf16, {order = #NHWC}>
    // CHECK:         [[TILE_11:%.*]] = VPU.Tile(%arg0) {repeats_values = [1, 1, 42, 512]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x42x512xf16, {order = #NHWC}>
    // CHECK:         [[RET:%.*]] = VPU.Concat([[TILE_0]], [[TILE_1]], [[TILE_2]], [[TILE_3]], [[TILE_4]], [[TILE_5]], [[TILE_6]], [[TILE_7]], [[TILE_8]], [[TILE_9]], [[TILE_10]], [[TILE_11]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 215, 0], [0, 0, 258, 0], [0, 0, 301, 0], [0, 0, 344, 0], [0, 0, 386, 0], [0, 0, 428, 0], [0, 0, 470, 0]
    // CHECK:         return [[RET]] : tensor<1x32x512x512xf16, {order = #NHWC}>
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

    // CHECK:       [[SLICE0:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 3, 227, 140]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x140xf16>

    // CHECK:       [[PAD0:%.*]] = VPU.Pad([[SLICE0]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x140xf16> -> tensor<1x16x227x150xf16>

    // Tile 1

    // CHECK:       [[SLICE1:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 140] [1, 3, 227, 150]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x150xf16>

    // CHECK:       [[PAD1:%.*]] = VPU.Pad([[SLICE1]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x150xf16> -> tensor<1x16x227x150xf16>

    // Tile 2

    // CHECK:       [[SLICE2:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 290] [1, 3, 227, 150]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x150xf16>

    // CHECK:       [[PAD2:%.*]] = VPU.Pad([[SLICE2]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]}
    // CHECK-SAME:      : tensor<1x3x227x150xf16> -> tensor<1x16x227x150xf16>

    // Tile 3

    // CHECK:       [[SLICE3:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 440] [1, 3, 227, 140]
    // CHECK-SAME:      : tensor<1x3x227x580xf16> to tensor<1x3x227x140xf16>

    // CHECK:       [[PAD3:%.*]] = VPU.Pad([[SLICE3]])
    // CHECK-SAME:      {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 10]}
    // CHECK-SAME:      : tensor<1x3x227x140xf16> -> tensor<1x16x227x150xf16>

    // Concat

    // CHECK:       [[CONCAT:%.*]] = VPU.Concat([[PAD0]], [[PAD1]], [[PAD2]], [[PAD3]]) {static_offsets =
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 150], [0, 0, 0, 300], [0, 0, 0, 450]
    // CHECK-SAME:      : tensor<1x16x227x150xf16>, tensor<1x16x227x150xf16>, tensor<1x16x227x150xf16>,
    // CHECK-SAME:      tensor<1x16x227x150xf16> -> tensor<1x16x227x600xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x16x227x600xf16>
}
