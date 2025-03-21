//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --tiling="enable-prefetch=false" %s | FileCheck %s
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
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x200x200xf16>)
            -> tensor<1x16x200x200xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } : tensor<1x16x200x200xf16> -> tensor<1x16x200x200xf16>
    return %1 : tensor<1x16x200x200xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16> to tensor<1x16x101x200xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x100x200xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func.func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:       : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Add([[INPUT_TILE0]], [[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16> to tensor<1x1024x14x14xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Add([[INPUT_TILE1]], [[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1024x14x14xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
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
// CHECK-SAME:      :  tensor<1x24x33x64xf16>
// CHECK-SAME:      -> tensor<1x24x128x256xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 31, 0] [1, 24, 33, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x24x33x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      :  tensor<1x24x33x64xf16>
// CHECK-SAME:      -> tensor<1x24x128x256xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 128, 0]]}
// CHECK-SAME:      : tensor<1x24x128x256xf16>, tensor<1x24x128x256xf16> -> tensor<1x24x256x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x4x360x640xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x4x360x640xf16>)
            -> tensor<1x4x144x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [144, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} :
        tensor<1x4x360x640xf16> -> tensor<1x4x144x256xf16>

    return %0 : tensor<1x4x144x256xf16>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 1] [1, 4, 360, 319]
// CHECK-SAME:      : tensor<1x4x360x640xf16> to tensor<1x4x360x319xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x4x360x319xf16>
// CHECK-SAME:      -> tensor<1x4x144x128xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 321] [1, 4, 360, 319]
// CHECK-SAME:      : tensor<1x4x360x640xf16> to tensor<1x4x360x319xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x4x360x319xf16>
// CHECK-SAME:      -> tensor<1x4x144x128xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 128]]}
// CHECK-SAME:      : tensor<1x4x144x128xf16>, tensor<1x4x144x128xf16> -> tensor<1x4x144x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x4x144x256xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1000x800xf16>
func.func @InterpSplitOverH(
        %input1: tensor<1x1x1000x800xf16>)
            -> tensor<1x1x860x620xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [860, 620], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} :
        tensor<1x1x1000x800xf16> -> tensor<1x1x860x620xf16>

    return %0 : tensor<1x1x860x620xf16>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 500, 800]
// CHECK-SAME:      : tensor<1x1x1000x800xf16> to tensor<1x1x500x800xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x1x500x800xf16>
// CHECK-SAME:      -> tensor<1x1x430x620xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 500, 0] [1, 1, 500, 800]
// CHECK-SAME:      : tensor<1x1x1000x800xf16> to tensor<1x1x500x800xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x1x500x800xf16>
// CHECK-SAME:      -> tensor<1x1x430x620xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 430, 0]]}
// CHECK-SAME:      : tensor<1x1x430x620xf16>, tensor<1x1x430x620xf16> -> tensor<1x1x860x620xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x1x860x620xf16>
}

// -----

// CHECK-LABEL: func.func @InterpCubicSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x4x360x640xf16>
func.func @InterpCubicSplitOverC(
        %input1: tensor<1x4x360x640xf16>)
            -> tensor<1x4x144x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01, mode = <CUBIC>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [144, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} :
        tensor<1x4x360x640xf16> -> tensor<1x4x144x256xf16>

    return %0 : tensor<1x4x144x256xf16>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 4, 360, 321]
// CHECK-SAME:      : tensor<1x4x360x640xf16> to tensor<1x4x360x321xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x4x360x321xf16>
// CHECK-SAME:      -> tensor<1x4x144x128xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 320] [1, 4, 360, 320]
// CHECK-SAME:      : tensor<1x4x360x640xf16> to tensor<1x4x360x320xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x4x360x320xf16>
// CHECK-SAME:      -> tensor<1x4x144x128xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 128]]}
// CHECK-SAME:      : tensor<1x4x144x128xf16>, tensor<1x4x144x128xf16> -> tensor<1x4x144x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x4x144x256xf16>
}

// -----

// CHECK-LABEL: func.func @InterpCubicSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1000x800xf16>
func.func @InterpCubicSplitOverH(
        %input1: tensor<1x1x1000x800xf16>)
            -> tensor<1x1x860x620xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01, mode = <CUBIC>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [860, 620], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} :
        tensor<1x1x1000x800xf16> -> tensor<1x1x860x620xf16>

    return %0 : tensor<1x1x860x620xf16>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 501, 800]
// CHECK-SAME:      : tensor<1x1x1000x800xf16> to tensor<1x1x501x800xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x1x501x800xf16>
// CHECK-SAME:      -> tensor<1x1x430x620xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 499, 0] [1, 1, 501, 800]
// CHECK-SAME:      : tensor<1x1x1000x800xf16> to tensor<1x1x501x800xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x1x501x800xf16>
// CHECK-SAME:      -> tensor<1x1x430x620xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 430, 0]]}
// CHECK-SAME:      : tensor<1x1x430x620xf16>, tensor<1x1x430x620xf16> -> tensor<1x1x860x620xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x1x860x620xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x24x64x64xf16>)
            -> tensor<1x24x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
        tensor<1x24x64x64xf16>-> tensor<1x24x256x256xf16>

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

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
func.func @InterpSplitOverH(
    %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
            -> tensor<1x64x192x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [192, 320],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
    return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>
}

// CHECK:  [[SLICE0:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 0] [1, 64, 48, 17] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x17xf16, {order = #NHWC}>
// CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[SLICE0]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE1:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 16] [1, 64, 48, 17] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x17xf16, {order = #NHWC}>
// CHECK:  [[INTERP1:%.+]] = VPU.Interpolate([[SLICE1]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE2:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 32] [1, 64, 48, 17] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x17xf16, {order = #NHWC}>
// CHECK:  [[INTERP2:%.+]] = VPU.Interpolate([[SLICE2]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE3:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 48] [1, 64, 48, 17] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x17xf16, {order = #NHWC}>
// CHECK:  [[INTERP3:%.+]] = VPU.Interpolate([[SLICE3]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE4:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 64] [1, 64, 48, 16] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x16xf16, {order = #NHWC}>
// CHECK:  [[INTERP4:%.+]] = VPU.Interpolate([[SLICE4]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]], [[INTERP3]], [[INTERP4]])
// CHECK:  return [[CONCAT]] : tensor<1x64x192x320xf16, {order = #NHWC}>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
func.func @InterpSplitOverCNoCommonFactor(
    %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
            -> tensor<1x64x121x121xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [121, 121],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x31x31xf16, {order = #NHWC}> -> tensor<1x64x121x121xf16, {order = #NHWC}>
    return %0 : tensor<1x64x121x121xf16, {order = #NHWC}>
}

// CHECK:  [[SLICE0:%.+]] = VPU.Slice %arg0
// CHECK-SAME:      [0, 0, 0, 0] [1, 64, 17, 31] : tensor<1x64x31x31xf16, {order = #NHWC}> to tensor<1x64x17x31xf16, {order = #NHWC}>
// CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[SLICE0]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 0, 15, 0]
// CHECK-SAME:      [1, 64, 16, 31] : tensor<1x64x31x31xf16, {order = #NHWC}> to tensor<1x64x16x31xf16, {order = #NHWC}>
// CHECK:  [[INTERP1:%.+]] = VPU.Interpolate([[SLICE1]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]])
// CHECK:  return [[CONCAT]] : tensor<1x64x121x121xf16, {order = #NHWC}>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NoTilingClusterNCEConv(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights_table as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1],
                ppe = #VPU.PPEStub<>
            } -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-DAG:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[NCE_CONV]]

    // CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @GatherSplit
func.func @GatherSplit(%arg0: tensor<4004x320xf16>) -> tensor<4004x1xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.CastElemType<si32>]
  %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<4004x1xf16>
  return %0 : tensor<4004x1xf16>

  // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.CastElemType<si32>]

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0] [2002, 320] : tensor<4004x320xf16> to tensor<2002x320xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<2002x320xf16>, tensor<1xsi32> -> tensor<2002x1xf16>

  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice %arg0 [2002, 0] [2002, 320] : tensor<4004x320xf16> to tensor<2002x320xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<2002x320xf16>, tensor<1xsi32> -> tensor<2002x1xf16>

  // CHECK:     [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0], [2002, 0]]} : tensor<2002x1xf16>, tensor<2002x1xf16> -> tensor<4004x1xf16>

  // CHECK:     return [[Concat]]
}

// -----

// CHECK-LABEL: func.func @GatherSplitWithBatchDims
func.func @GatherSplitWithBatchDims(%arg0: tensor<2x4004x320xf16>) -> tensor<2x1x320xf16> {
  %cst = const.Declare tensor<2x1xsi32> = dense<[[-4004], [4003]]> : tensor<2x1xsi64>, [#const.CastElemType<si32>]
  %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>
  return %0 : tensor<2x1x320xf16>

  // CHECK:     [[CST:%.+]] = const.Declare tensor<1x1xsi32>
  // CHECK-SAME:tensor<2x1xsi64>, [#const.SubView<[1, 0], [1, 1]>, #const.CastElemType<si32>]
  // CHECK:     [[CST0:%.+]] = const.Declare tensor<1x1xsi32>
  // CHECK-SAME:tensor<2x1xsi64>, [#const.SubView<[0, 0], [1, 1]>, #const.CastElemType<si32>]

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0, 0] [1, 4004, 160] : tensor<2x4004x320xf16> to tensor<1x4004x160xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[CST0]]) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<1x4004x160xf16>, tensor<1x1xsi32> -> tensor<1x1x160xf16>

  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice %arg0 [0, 0, 160] [1, 4004, 160] : tensor<2x4004x320xf16> to tensor<1x4004x160xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[CST0]]) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<1x4004x160xf16>, tensor<1x1xsi32> -> tensor<1x1x160xf16>

  // Tile 2
  // CHECK:     [[Tile2:%.+]] = VPU.Slice %arg0 [1, 0, 0] [1, 4004, 160] : tensor<2x4004x320xf16> to tensor<1x4004x160xf16>
  // CHECK:     [[Gather2:%.+]] = VPU.Gather([[Tile2]], [[CST]]) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<1x4004x160xf16>, tensor<1x1xsi32> -> tensor<1x1x160xf16>

  // Tile 3
  // CHECK:     [[Tile3:%.+]] = VPU.Slice %arg0 [1, 0, 160] [1, 4004, 160] : tensor<2x4004x320xf16> to tensor<1x4004x160xf16>
  // CHECK:     [[Gather3:%.+]] = VPU.Gather([[Tile3]], [[CST]]) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<1x4004x160xf16>, tensor<1x1xsi32> -> tensor<1x1x160xf16>

  // CHECK:    [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]], [[Gather2]], [[Gather3]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 160], [1, 0, 0], [1, 0, 160]]} : tensor<1x1x160xf16>, tensor<1x1x160xf16>, tensor<1x1x160xf16>, tensor<1x1x160xf16> -> tensor<2x1x320xf16>

  // CHECK:     return [[Concat]] : tensor<2x1x320xf16>
}

// -----

// CHECK-LABEL: func.func @GatherSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<387072x3xf16>
func.func @GatherSplitOptimize(%arg0: tensor<387072x3xf16>) -> tensor<1x387072x3xf16> {
  %cst = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.CastElemType<si32>]
  %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>
  return %0 : tensor<1x387072x3xf16>

  // CHECK:     [[CST:%.+]] = const.Declare tensor<1x193536xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 193536], [1, 193536]>, #const.CastElemType<si32>]
  // CHECK:     [[CST0:%.+]] = const.Declare tensor<1x193536xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 0], [1, 193536]>, #const.CastElemType<si32>]

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // Tile 2
  // CHECK:     [[Tile2:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather2:%.+]] = VPU.Gather([[Tile2]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // Tile 3
  // CHECK:     [[Tile3:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather3:%.+]] = VPU.Gather([[Tile3]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // Tile 4
  // CHECK:     [[Tile4:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather4:%.+]] = VPU.Gather([[Tile4]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // Tile 5
  // CHECK:     [[Tile5:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather5:%.+]] = VPU.Gather([[Tile5]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x193536xsi32> -> tensor<1x193536x1xf16>

  // CHECK:    [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]], [[Gather2]], [[Gather3]], [[Gather4]], [[Gather5]])
  // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 193536, 0], [0, 193536, 1], [0, 193536, 2]]} : tensor<1x193536x1xf16>, tensor<1x193536x1xf16>, tensor<1x193536x1xf16>, tensor<1x193536x1xf16>, tensor<1x193536x1xf16>, tensor<1x193536x1xf16> -> tensor<1x387072x3xf16>

  // CHECK:     return [[Concat]] : tensor<1x387072x3xf16>
}

// -----

// CHECK-LABEL: func.func @Gather4DSplit
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x4004x320x1xf16>
func.func @Gather4DSplit(%arg0: tensor<1x4004x320x1xf16>) -> tensor<1x4004x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<4003> : tensor<1x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x4004x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x4004x1x1xf16>
    return %0 : tensor<1x4004x1x1xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<4003> : tensor<1x1x1x1xsi32>
    // CHECK:     [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2002, 320, 1] : tensor<1x4004x320x1xf16> to tensor<1x2002x320x1xf16>
    // CHECK:     [[GATHER0:%.+]] = VPU.Gather([[SLICE0]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x2002x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x2002x1x1xf16>

    // CHECK:     [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2002, 0, 0] [1, 2002, 320, 1] : tensor<1x4004x320x1xf16> to tensor<1x2002x320x1xf16>
    // CHECK:     [[GATHER1:%.+]] = VPU.Gather([[SLICE1]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x2002x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x2002x1x1xf16>
    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]]) {
    // CHECK-SAME{LITERAL}:             static_offsets = [[0, 0, 0, 0], [0, 2002, 0, 0]]} : tensor<1x2002x1x1xf16>, tensor<1x2002x1x1xf16> -> tensor<1x4004x1x1xf16>

    // CHECK:     return [[CONCAT]]
}

// -----

// CHECK-LABEL: func.func @Gather4DSplitWithBatchDims
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x1x4004x320xf16>
func.func @Gather4DSplitWithBatchDims(%arg0: tensor<2x1x4004x320xf16>) -> tensor<2x1x1x320xf16> {
    %cst = const.Declare tensor<2x1x1x1xsi32> = dense<1> : tensor<2x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<2x1x4004x320xf16>, tensor<2x1x1x1xsi32> -> tensor<2x1x1x320xf16>
    return %0 : tensor<2x1x1x320xf16>

    // CHECK-DAG: [[INDICES0:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<2x1x1x1xsi32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 1, 1]>]
    // CHECK-DAG: [[INDICES1:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<2x1x1x1xsi32>, [#const.SubView<[1, 0, 0, 0], [1, 1, 1, 1]>]

    // CHECK:     [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 4004, 160] : tensor<2x1x4004x320xf16> to tensor<1x1x4004x160xf16>
    // CHECK:     [[GATHER0:%.+]] = VPU.Gather([[SLICE0]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x4004x160xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x160xf16>

    // CHECK:     [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 160] [1, 1, 4004, 160] : tensor<2x1x4004x320xf16> to tensor<1x1x4004x160xf16>
    // CHECK:     [[GATHER1:%.+]] = VPU.Gather([[SLICE1]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x4004x160xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x160xf16>

    // CHECK:     [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 0] [1, 1, 4004, 160] : tensor<2x1x4004x320xf16> to tensor<1x1x4004x160xf16>
    // CHECK:     [[GATHER2:%.+]] = VPU.Gather([[SLICE2]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x4004x160xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x160xf16>

    // CHECK:     [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [1, 0, 0, 160] [1, 1, 4004, 160] : tensor<2x1x4004x320xf16> to tensor<1x1x4004x160xf16>
    // CHECK:     [[GATHER3:%.+]] = VPU.Gather([[SLICE3]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x4004x160xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x160xf16>

    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]], [[GATHER2]], [[GATHER3]]) {
    // CHECK-SAME{LITERAL}:             static_offsets = [[0, 0, 0, 0], [0, 0, 0, 160], [1, 0, 0, 0], [1, 0, 0, 160]]}

    // CHECK:     return [[CONCAT]]
}

// -----

// CHECK-LABEL: func.func @Gather4DSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x387072x3xf16>
func.func @Gather4DSplitOptimize(%arg0: tensor<1x1x387072x3xf16>) -> tensor<1x1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x3xf16>, tensor<1x387072x1x1xsi32> -> tensor<1x1x387072x3xf16>
    return %0 : tensor<1x1x387072x3xf16>

    // CHECK-DAG: [[INDICES0:%.+]] = const.Declare tensor<1x193536x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>, [#const.SubView<[0, 0, 0, 0], [1, 193536, 1, 1]>]
    // CHECK-DAG: [[INDICES1:%.+]] = const.Declare tensor<1x193536x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>, [#const.SubView<[0, 193536, 0, 0], [1, 193536, 1, 1]>]

    // CHECK:     [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER0:%.+]] = VPU.Gather([[SLICE0]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER1:%.+]] = VPU.Gather([[SLICE1]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER2:%.+]] = VPU.Gather([[SLICE2]], [[INDICES0]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[SLICE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER3:%.+]] = VPU.Gather([[SLICE3]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[SLICE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER4:%.+]] = VPU.Gather([[SLICE4]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[SLICE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2] [1, 1, 387072, 1] : tensor<1x1x387072x3xf16> to tensor<1x1x387072x1xf16>
    // CHECK:     [[GATHER5:%.+]] = VPU.Gather([[SLICE5]], [[INDICES1]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x1xf16>, tensor<1x193536x1x1xsi32> -> tensor<1x1x193536x1xf16>

    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]], [[GATHER2]], [[GATHER3]], [[GATHER4]], [[GATHER5]]) {
    // CHECK-SAME{LITERAL}:             static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 193536, 0], [0, 0, 193536, 1], [0, 0, 193536, 2]]

    // CHECK:     return [[CONCAT]]
}

// -----

// CHECK-LABEL: func.func @Yuv2RGBSplit
func.func @Yuv2RGBSplit(%arg0: tensor<1x993x982x1xf16>) -> tensor<1x662x982x3xf16> {
  %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
  %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
  %2 = VPU.Reshape(%1) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
  %3 = VPU.YuvToRgb(%0, %2) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
  return %3 : tensor<1x662x982x3xf16>

    // CHECK:    %0 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    // CHECK:    %1 = VPU.ShapeCast {shape = [1, 331, 491, 2]} inputs(%0 : tensor<1x331x982x1xf16>) -> tensor<1x331x491x2xf16>

    // Tile 0
    // CHECK:    %2 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %3 = VPU.Slice %1 [0, 0, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %4 = VPU.YuvToRgb(%2, %3) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 1
    // CHECK:    %5 = VPU.Slice %arg0 [0, 220, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %6 = VPU.Slice %1 [0, 110, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %7 = VPU.YuvToRgb(%5, %6) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 2
    // CHECK:    %8 = VPU.Slice %arg0 [0, 440, 0, 0] [1, 220, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x220x982x1xf16>
    // CHECK:    %9 = VPU.Slice %1 [0, 220, 0, 0] [1, 110, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x110x491x2xf16>
    // CHECK:    %10 = VPU.YuvToRgb(%8, %9) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x220x982x1xf16>, tensor<1x110x491x2xf16> -> tensor<1x220x982x3xf16>

    // Tile 3
    // CHECK:    %11 = VPU.Slice %arg0 [0, 660, 0, 0] [1, 2, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x2x982x1xf16>
    // CHECK:    %12 = VPU.Slice %1 [0, 330, 0, 0] [1, 1, 491, 2] : tensor<1x331x491x2xf16> to tensor<1x1x491x2xf16>
    // CHECK:    %13 = VPU.YuvToRgb(%11, %12) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x2x982x1xf16>, tensor<1x1x491x2xf16> -> tensor<1x2x982x3xf16>

    // CHECK:    %14 = VPU.Concat(%4, %7, %10, %13)
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 220, 0, 0], [0, 440, 0, 0], [0, 660, 0, 0]]} : tensor<1x220x982x3xf16>, tensor<1x220x982x3xf16>, tensor<1x220x982x3xf16>, tensor<1x2x982x3xf16> -> tensor<1x662x982x3xf16>
    // CHECK:    return %14 : tensor<1x662x982x3xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplit
func.func @GatherNDSplit(%arg0: tensor<3x5x512x512xf16>) -> tensor<3x1x100x512xf16> {
    %cst = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>
    return %0 : tensor<3x1x100x512xf16>

    // CHECK-DAG: [[Indices_2:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[2, 0, 0, 0], [1, 1, 100, 2]>]
    // CHECK-DAG: [[Indices_1:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[1, 0, 0, 0], [1, 1, 100, 2]>]
    // CHECK-DAG: [[Indices_0:%.+]] = const.Declare tensor<1x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>, [#const.SubView<[0, 0, 0, 0], [1, 1, 100, 2]>]

    // CHECK: [[Tile0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND0:%.+]] = VPU.GatherND([[Tile0]], [[Indices_0]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND1:%.+]] = VPU.GatherND([[Tile1]], [[Indices_0]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile2:%.+]] = VPU.Slice %arg0 [1, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND2:%.+]] = VPU.GatherND([[Tile2]], [[Indices_1]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile3:%.+]] = VPU.Slice %arg0 [1, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND3:%.+]] = VPU.GatherND([[Tile3]], [[Indices_1]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile4:%.+]] = VPU.Slice %arg0 [2, 0, 0, 0] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND4:%.+]] = VPU.GatherND([[Tile4]], [[Indices_2]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[Tile5:%.+]] = VPU.Slice %arg0 [2, 0, 0, 256] [1, 5, 512, 256] : tensor<3x5x512x512xf16> to tensor<1x5x512x256xf16>
    // CHECK: [[GatherND5:%.+]] = VPU.GatherND([[Tile5]], [[Indices_2]]) {batch_dims = 1 : i64} : tensor<1x5x512x256xf16>, tensor<1x1x100x2xsi32> -> tensor<1x1x100x256xf16>

    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[GatherND0]], [[GatherND1]], [[GatherND2]], [[GatherND3]], [[GatherND4]], [[GatherND5]])
    // CHECK-SAME: [0, 0, 0, 0], [0, 0, 0, 256], [1, 0, 0, 0], [1, 0, 0, 256], [2, 0, 0, 0], [2, 0, 0, 256]
    // CHECK-SAME: -> tensor<3x1x100x512xf16>

    // CHECK: return [[OUTPUT]] : tensor<3x1x100x512xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitIndices
func.func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<400000x2xf16> {
    %cst = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>
    return %0 : tensor<400000x2xf16>

    // CHECK-DAG: [[Indices_1:%.+]] = const.Declare tensor<200000x1xsi32> = dense<1> : tensor<400000x1xsi32>, [#const.SubView<[200000, 0], [200000, 1]>]
    // CHECK-DAG: [[Indices_0:%.+]] = const.Declare tensor<200000x1xsi32> = dense<1> : tensor<400000x1xsi32>, [#const.SubView<[0, 0], [200000, 1]>]

    // CHECK: [[GatherND0:%.+]] = VPU.GatherND(%arg0, [[Indices_0]]) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<200000x1xsi32> -> tensor<200000x2xf16>
    // CHECK: [[GatherND1:%.+]] = VPU.GatherND(%arg0, [[Indices_1]]) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<200000x1xsi32> -> tensor<200000x2xf16>

    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[GatherND0]], [[GatherND1]])
    // CHECK-SAME: [0, 0], [200000, 0]
    // CHECK-SAME: -> tensor<400000x2xf16>

    // CHECK: return [[OUTPUT]] : tensor<400000x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceBlocksFirstSplit
func.func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
  return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x240x10x120xf32, {order = #NHWC}>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x240x10x120xf32, {order = #NHWC}> -> tensor<1x240x10x120xf16, {order = #NHWC}>

  // CHECK:     %2 = VPU.Slice %arg0 [0, 240, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x240x10x120xf32, {order = #NHWC}>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x240x10x120xf32, {order = #NHWC}> -> tensor<1x240x10x120xf16, {order = #NHWC}>

  // CHECK:     %4 = VPU.Concat(%1, %3) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 240, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16, {order = #NHWC}>

  // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %6 = VPU.DepthToSpace(%5) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %7 = VPU.Slice %4 [0, 0, 5, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %9 = VPU.Concat(%6, %8)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16, {order = #NHWC}>

  // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 30, 40, 240] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x240xf16, {order = #NHWC}>
  // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32} : tensor<1x30x40x240xf16, {order = #NHWC}> -> tensor<1x30x40x240xf32, {order = #NHWC}>

  // CHECK:     %12 = VPU.Slice %9 [0, 0, 0, 240] [1, 30, 40, 240] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x240xf16, {order = #NHWC}>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x30x40x240xf16, {order = #NHWC}> -> tensor<1x30x40x240xf32, {order = #NHWC}>

  // CHECK:     %14 = VPU.Concat(%11, %13)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 240]
  // CHECK-SAME:-> tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     return %14 : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplit
func.func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
  return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x240x10x120xf32, {order = #NHWC}>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x240x10x120xf32, {order = #NHWC}> -> tensor<1x240x10x120xf16, {order = #NHWC}>

  // CHECK:     %2 = VPU.Slice %arg0 [0, 240, 0, 0] [1, 240, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x240x10x120xf32, {order = #NHWC}>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x240x10x120xf32, {order = #NHWC}> -> tensor<1x240x10x120xf16, {order = #NHWC}>

  // CHECK:     %4 = VPU.Concat(%1, %3) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 240, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16, {order = #NHWC}>

  // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %6 = VPU.DepthToSpace(%5) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %7 = VPU.Slice %4 [0, 0, 5, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %9 = VPU.Concat(%6, %8)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16, {order = #NHWC}>

  // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 30, 40, 240] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x240xf16, {order = #NHWC}>
  // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32} : tensor<1x30x40x240xf16, {order = #NHWC}> -> tensor<1x30x40x240xf32, {order = #NHWC}>

  // CHECK:     %12 = VPU.Slice %9 [0, 0, 0, 240] [1, 30, 40, 240] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x240xf16, {order = #NHWC}>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x30x40x240xf16, {order = #NHWC}> -> tensor<1x30x40x240xf32, {order = #NHWC}>

  // CHECK:     %14 = VPU.Concat(%11, %13)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 240]
  // CHECK-SAME:-> tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     return %14 : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @SpaceToDepthBlockFirstSplit
func.func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
      %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
      %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
      %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
      return %2 : tensor<1x768x40x20xf32>

      // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
      // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

      // CHECK:     %2 = VPU.Slice %arg0 [0, 0, 80, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
      // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

      // CHECK:     %4 = VPU.Concat(%1, %3)
      // CHECK-SAME:[0, 0, 0, 0], [0, 0, 80, 0]
      // CHECK-SAME:-> tensor<1x48x160x80xf16>

      // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 48, 160, 40]  : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
      // CHECK:     %6 = VPU.SpaceToDepthOp(%5) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

      // CHECK:     %7 = VPU.Slice %4 [0, 0, 0, 40]  [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
      // CHECK:     %8 = VPU.SpaceToDepthOp(%7) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

      // CHECK:     %9 = VPU.Concat(%6, %8)
      // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 10]
      // CHECK-SAME:-> tensor<1x768x40x20xf16>

      // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
      // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

      // CHECK:     %12 = VPU.Slice %9 [0, 384, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
      // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

      // CHECK:     %14 = VPU.Concat(%11, %13)
      // CHECK-SAME:[0, 0, 0, 0], [0, 384, 0, 0]
      // CHECK-SAME:-> tensor<1x768x40x20xf32>

      // CHECK:     return %14 : tensor<1x768x40x20xf32>
}

// -----

// CHECK-LABEL: func.func @SpaceToDepthDepthFirstSplit
func.func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
    // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

    // CHECK:     %2 = VPU.Slice %arg0 [0, 0, 80, 0] [1, 48, 80, 80] : tensor<1x48x160x80xf32> to tensor<1x48x80x80xf32>
    // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x48x80x80xf32> -> tensor<1x48x80x80xf16>

    // CHECK:     %4 = VPU.Concat(%1, %3)
    // CHECK-SAME:[0, 0, 0, 0], [0, 0, 80, 0]
    // CHECK-SAME:-> tensor<1x48x160x80xf16>

    // CHECK:     %5 = VPU.Slice %4 [0, 0, 0, 0] [1, 48, 160, 40]  : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %6 = VPU.SpaceToDepthOp(%5) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:     %7 = VPU.Slice %4 [0, 0, 0, 40]  [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %8 = VPU.SpaceToDepthOp(%7) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>

    // CHECK:     %9 = VPU.Concat(%6, %8)
    // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 10]
    // CHECK-SAME:-> tensor<1x768x40x20xf16>

    // CHECK:     %10 = VPU.Slice %9 [0, 0, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
    // CHECK:     %11 = VPU.Convert(%10) {dstElemType = f32} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

    // CHECK:     %12 = VPU.Slice %9 [0, 384, 0, 0] [1, 384, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x384x40x20xf16>
    // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x384x40x20xf16> -> tensor<1x384x40x20xf32>

    // CHECK:     %14 = VPU.Concat(%11, %13)
    // CHECK-SAME:[0, 0, 0, 0], [0, 384, 0, 0]
    // CHECK-SAME:-> tensor<1x768x40x20xf32>

    // CHECK:     return %14 : tensor<1x768x40x20xf32>
  }

// -----




#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

// CHECK:    [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<512x32x3x3x!qElemType2,
// CHECK-SAME:    {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
// CHECK:    [[FILTER_TILE:%.+]] = const.Declare tensor<512x1x1x4xsi32,
// CHECK-SAME:  {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>
// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
// CHECK-SAME:  tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x32x33x64x!qElemType, {order = #NHWC}>
// CHECK:    [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS_TABLE_TILE]], [[FILTER_TILE]])
// CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
// CHECK-SAME:  rawFilterShape = [512, 32, 3, 3], strides = [1, 1]}
// CHECK-SAME:  -> tensor<1x512x32x64x!qElemType1, {order = #NHWC}>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
// CHECK-SAME:   tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x32x33x64x!qElemType, {order = #NHWC}>
// CHECK:    [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS_TABLE_TILE]], [[FILTER_TILE]])
// CHECK-SAME:  pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
// CHECK-SAME:  rawFilterShape = [512, 32, 3, 3], strides = [1, 1]}
// CHECK-SAME:  -> tensor<1x512x32x64x!qElemType1, {order = #NHWC}>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]])
// CHECK-LITERAL:  {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}
// CHECK-SAME:   tensor<1x512x32x64x!qElemType1, {order = #NHWC}>, tensor<1x512x32x64x!qElemType1, {order = #NHWC}>
// CHECK-SAME:  -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
// CHECK:    return [[CONCAT]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 99, 0] [1, 16, 101, 200]
    // CHECK-SAME:      : tensor<1x16x200x200xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x101x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      } -> tensor<1x16x100x200xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 100, 0]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x24xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x24xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x24xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x24xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 512, 0, 0] [1, 512, 24, 24]
    // CHECK-SAME:      : tensor<1x1024x24x24xf16, {order = #NHWC}> to tensor<1x512x24x24xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x512x24x24xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:      -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE0]], [[INPUT_TILE0]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 1024, 0, 0] [1, 1024, 14, 14]
    // CHECK-SAME:      : tensor<1x2048x14x14xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT_TILE1]], [[INPUT_TILE1]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      } -> tensor<1x1024x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 1024, 0, 0]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ConvertU8F32SplitOverW(%arg0: tensor<1x2x80x4000xui8, {order = #NHWC}>) -> tensor<1x2x80x4000xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x4000xui8, {order = #NHWC}> -> tensor<1x2x80x4000xf32, {order = #NHWC}>
  return %0 : tensor<1x2x80x4000xf32, {order = #NHWC}>
}

// CHECK-LABEL: @ConvertU8F32SplitOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x4000xui8, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 80, 2000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x80x2000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convert([[INPUT_TILE0]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x80x2000xf32, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2000] [1, 2, 80, 2000]
// CHECK-SAME:      : tensor<1x2x80x4000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x80x2000xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convert([[INPUT_TILE1]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x80x2000xf32, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 2000]
// CHECK-SAME:      -> tensor<1x2x80x4000xf32, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x2x80x4000xf32, {order = #NHWC}>

// -----

func.func @SigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sigmoid([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sigmoid([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @TanhSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tanh(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @TanhSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tanh([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tanh([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @ExpSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Exp(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ExpSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Exp([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Exp([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SqrtSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SqrtSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sqrt([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sqrt([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
// -----

func.func @EluSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @EluSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Elu([[INPUT_TILE0]]) {
// CHECK-SAME:    x = 1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Elu([[INPUT_TILE1]]) {
// CHECK-SAME:    x = 1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:  } : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:  } : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

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

// -----

func.func @HSwishSplitOverW(%arg0: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {
  %0 = VPU.HSwish(%arg0) : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>
  return %0 : tensor<1x16x80x1280xf16>
}

// CHECK-LABEL: @HSwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 80, 320]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x80x320xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSwish([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x16x80x320xf16> -> tensor<1x16x80x320xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 320] [1, 16, 80, 320]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x80x320xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSwish([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x16x80x320xf16> -> tensor<1x16x80x320xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 16, 80, 320]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x80x320xf16>
// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.HSwish([[INPUT_TILE2]])
// CHECK-SAME:  : tensor<1x16x80x320xf16> -> tensor<1x16x80x320xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 960] [1, 16, 80, 320]
// CHECK-SAME:  : tensor<1x16x80x1280xf16> to tensor<1x16x80x320xf16>
// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.HSwish([[INPUT_TILE3]])
// CHECK-SAME:  : tensor<1x16x80x320xf16> -> tensor<1x16x80x320xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 320], [0, 0, 0, 640], [0, 0, 0, 960]
// CHECK-SAME:  : tensor<1x16x80x320xf16>, tensor<1x16x80x320xf16>, tensor<1x16x80x320xf16>,
// CHECK-SAME:   tensor<1x16x80x320xf16> -> tensor<1x16x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x80x1280xf16>

// -----

func.func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Divide([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Divide([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

// CHECK-LABEL: func.func @SplitFakeQuant
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x80x80xf16>
func.func @SplitFakeQuant(%arg0: tensor<1x128x80x80xf16>) -> tensor<1x128x80x80xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<10.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
    %cst_2 = const.Declare tensor<1x1x1x1xf16> = dense<11.000000e+00> : tensor<1x1x1x1xf16>
    %0 = VPU.FakeQuantize(%arg0, %cst, %cst_0, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<1x128x80x80xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x80x80xf16>
    return %0 : tensor<1x128x80x80xf16>

    // Tile 0
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 80, 80]
    // CHECK-SAME:  : tensor<1x128x80x80xf16> to tensor<1x64x80x80xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.FakeQuantize([[INPUT_TILE0]], %cst, %cst_0, %cst_1, %cst_2)
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:       : tensor<1x64x80x80xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x80x80xf16>

    // Tile 1
    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 64, 0, 0] [1, 64, 80, 80]
    // CHECK-SAME:  : tensor<1x128x80x80xf16> to tensor<1x64x80x80xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.FakeQuantize([[INPUT_TILE1]], %cst, %cst_0, %cst_1, %cst_2)
    // CHECK-SAME:       {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64}
    // CHECK-SAME:       : tensor<1x64x80x80xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x80x80xf16>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 64, 0, 0]
    // CHECK-SAME:  :  tensor<1x64x80x80xf16>, tensor<1x64x80x80xf16> -> tensor<1x128x80x80xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x80x80xf16>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0035290287990196079>

// CHECK-LABEL: func.func @SplitQuantize
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x256x64x80xf16>
func.func @SplitQuantize(%arg0: tensor<1x256x64x80xf16>) -> tensor<1x256x64x80x!qElemType> {
    %0 = VPU.Quantize(%arg0) {dstElemType = !qElemType} : tensor<1x256x64x80xf16> -> tensor<1x256x64x80x!qElemType>
    return %0 : tensor<1x256x64x80x!qElemType>

    // Tile 0
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 256, 32, 80]
    // CHECK-SAME:  : tensor<1x256x64x80xf16> to tensor<1x256x32x80xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Quantize([[INPUT_TILE0]]) {dstElemType = !qElemType}
    // CHECK-SAME:  : tensor<1x256x32x80xf16> -> tensor<1x256x32x80x!qElemType>

    // Tile 1
    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 32, 0] [1, 256, 32, 80]
    // CHECK-SAME:  : tensor<1x256x64x80xf16> to tensor<1x256x32x80xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Quantize([[INPUT_TILE1]]) {dstElemType = !qElemType}
    // CHECK-SAME:  : tensor<1x256x32x80xf16> -> tensor<1x256x32x80x!qElemType>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:  :  tensor<1x256x32x80x!qElemType>, tensor<1x256x32x80x!qElemType> -> tensor<1x256x64x80x!qElemType>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x80x!qElemType>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0035290287990196079>

// CHECK-LABEL: func.func @SplitDequantize
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x256x64x80x!qElemType>
func.func @SplitDequantize(%arg0: tensor<1x256x64x80x!qElemType>) -> tensor<1x256x64x80xf16> {
    %0 = VPU.Dequantize(%arg0) {dstElemType = f16} : tensor<1x256x64x80x!qElemType> -> tensor<1x256x64x80xf16>
    return %0 : tensor<1x256x64x80xf16>

    // Tile 0
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 128, 64, 80]
    // CHECK-SAME:  : tensor<1x256x64x80x!qElemType> to tensor<1x128x64x80x!qElemType>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Dequantize([[INPUT_TILE0]]) {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x128x64x80x!qElemType> -> tensor<1x128x64x80xf16>

    // Tile 1
    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 128, 0, 0] [1, 128, 64, 80]
    // CHECK-SAME:  : tensor<1x256x64x80x!qElemType> to tensor<1x128x64x80x!qElemType>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Dequantize([[INPUT_TILE1]]) {dstElemType = f16}
    // CHECK-SAME:  : tensor<1x128x64x80x!qElemType> -> tensor<1x128x64x80xf16>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:  :  tensor<1x128x64x80xf16>, tensor<1x128x64x80xf16> -> tensor<1x256x64x80xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x80xf16>
}

// -----

// CHECK-LABEL: func.func @SplitMvn6OverW
// CHECK-SAME:        [[INPUT:%arg0]]: tensor<1x1x512x1200xf16>
func.func @SplitMvn6OverW(%arg0: tensor<1x1x512x1200xf16>) -> tensor<1x1x512x1200xf16> {
    %0 = VPU.MVN6(%arg0) {axes = [2], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>} : tensor<1x1x512x1200xf16> -> tensor<1x1x512x1200xf16>
    return %0 : tensor<1x1x512x1200xf16>

    // Tile 0
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 512, 600]
    // CHECK-SAME:  : tensor<1x1x512x1200xf16> to tensor<1x1x512x600xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MVN6([[INPUT_TILE0]]) {axes = [2], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK-SAME:  : tensor<1x1x512x600xf16> -> tensor<1x1x512x600xf16>

    // Tile 1
    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 600] [1, 1, 512, 600]
    // CHECK-SAME:  : tensor<1x1x512x1200xf16> to tensor<1x1x512x600xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MVN6([[INPUT_TILE1]]) {axes = [2], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 0, 0>}
    // CHECK-SAME:  : tensor<1x1x512x600xf16> -> tensor<1x1x512x600xf16>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 600]
    // CHECK-SAME:  :  tensor<1x1x512x600xf16>, tensor<1x1x512x600xf16> -> tensor<1x1x512x1200xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x512x1200xf16>
}

// -----

// CHECK-LABEL: func.func @SplitMvn6OverH
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x1500x512xf16>, [[SCALE:%.+]]: tensor<1x1x1x512xf16>, [[BIAS:%.+]]: tensor<1x1x1x512xf16>
func.func @SplitMvn6OverH(%arg0: tensor<1x1x1500x512xf16>, %arg1: tensor<1x1x1x512xf16>, %arg2: tensor<1x1x1x512xf16>) -> tensor<1x1x1500x512xf16> {
    %0 = VPU.MVN6(%arg0, %arg1, %arg2) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x1x1500x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1500x512xf16>
    return %0 : tensor<1x1x1500x512xf16>

    // Tile 0
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 750, 512]
    // CHECK-SAME:  : tensor<1x1x1500x512xf16> to tensor<1x1x750x512xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MVN6([[INPUT_TILE0]], [[SCALE]], [[BIAS]]) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:  : tensor<1x1x750x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x750x512xf16>

    // Tile 1
    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 750, 0] [1, 1, 750, 512]
    // CHECK-SAME:  : tensor<1x1x1500x512xf16> to tensor<1x1x750x512xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MVN6([[INPUT_TILE1]], [[SCALE]], [[BIAS]]) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:  : tensor<1x1x750x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x750x512xf16>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 750, 0]
    // CHECK-SAME:  :  tensor<1x1x750x512xf16>, tensor<1x1x750x512xf16> -> tensor<1x1x1500x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1500x512xf16>
}

// -----

// CHECK-LABEL: func.func @SplitMvn6OverHall
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x1500x512xf16>, [[SCALE:%.+]]: tensor<1x1x1500x1xf16>, [[BIAS:%.+]]: tensor<1x1x1500x1xf16>
func.func @SplitMvn6OverHall(%arg0: tensor<1x1x1500x512xf16>, %arg1: tensor<1x1x1500x1xf16>, %arg2: tensor<1x1x1500x1xf16>) -> tensor<1x1x1500x512xf16> {
    %0 = VPU.MVN6(%arg0, %arg1, %arg2) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>} : tensor<1x1x1500x512xf16>, tensor<1x1x1500x1xf16>, tensor<1x1x1500x1xf16> -> tensor<1x1x1500x512xf16>
    return %0 : tensor<1x1x1500x512xf16>

    // Tile 0
    // CHECK:       [[INPUT_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 750, 512] : tensor<1x1x1500x512xf16> to tensor<1x1x750x512xf16>
    // CHECK:       [[SCALE_TILE_0:%.+]] = VPU.Slice [[SCALE]] [0, 0, 0, 0] [1, 1, 750, 1] : tensor<1x1x1500x1xf16> to tensor<1x1x750x1xf16>
    // CHECK:       [[BIAS_TILE_0:%.+]]  = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 1, 750, 1] : tensor<1x1x1500x1xf16> to tensor<1x1x750x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MVN6([[INPUT_TILE_0]], [[SCALE_TILE_0]], [[BIAS_TILE_0]]) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:  : tensor<1x1x750x512xf16>, tensor<1x1x750x1xf16>, tensor<1x1x750x1xf16> -> tensor<1x1x750x512xf16>

    // Tile 1
    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 750, 0] [1, 1, 750, 512] : tensor<1x1x1500x512xf16> to tensor<1x1x750x512xf16>
    // CHECK:       [[SCALE_TILE_1:%.+]] = VPU.Slice [[SCALE]] [0, 0, 750, 0] [1, 1, 750, 1] : tensor<1x1x1500x1xf16> to tensor<1x1x750x1xf16>
    // CHECK:       [[BIAS_TILE_1:%.+]]  = VPU.Slice [[BIAS]] [0, 0, 750, 0] [1, 1, 750, 1] : tensor<1x1x1500x1xf16> to tensor<1x1x750x1xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MVN6([[INPUT_TILE_1]], [[SCALE_TILE_1]], [[BIAS_TILE_1]]) {axes = [3], eps = 2.500000e+00 : f64, eps_mode = #IE.mvn_eps_mode<INSIDE_SQRT>, normalize_variance = true, operandSegmentSizes = array<i32: 1, 1, 1>}
    // CHECK-SAME:  : tensor<1x1x750x512xf16>, tensor<1x1x750x1xf16>, tensor<1x1x750x1xf16> -> tensor<1x1x750x512xf16>

    // Concat
    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 750, 0]
    // CHECK-SAME:  :  tensor<1x1x750x512xf16>, tensor<1x1x750x512xf16> -> tensor<1x1x1500x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1500x512xf16>
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func.func @MemPermuteSplit3D2Part(%arg0: tensor<546x40x40xf16>) -> tensor<40x40x546xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, mem_perm = affine_map<(d0, d1, d2) -> (d1, d2, d0)>} : tensor<546x40x40xf16> -> tensor<40x40x546xf16>
  return %0 : tensor<40x40x546xf16>
}
// CHECK-LABEL: @MemPermuteSplit3D2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<546x40x40xf16>) -> tensor<40x40x546xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0] [273, 40, 40]
// CHECK-SAME:  : tensor<546x40x40xf16> to tensor<273x40x40xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #CHW, mem_perm = #HWC
// CHECK-SAME:  } : tensor<273x40x40xf16> -> tensor<40x40x273xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [273, 0, 0] [273, 40, 40]
// CHECK-SAME:  : tensor<546x40x40xf16> to tensor<273x40x40xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #CHW, mem_perm = #HWC
// CHECK-SAME:  } : tensor<273x40x40xf16> -> tensor<40x40x273xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0], [0, 0, 273]
// CHECK-SAME:  : tensor<40x40x273xf16>, tensor<40x40x273xf16> -> tensor<40x40x546xf16>

// CHECK:       return [[OUTPUT]] : tensor<40x40x546xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>
  return %0 : tensor<1x40x40x546xf16>
}
// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 273, 40, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x273x40x40xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC
// CHECK-SAME:  } : tensor<1x273x40x40xf16> -> tensor<1x40x40x273xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 273, 0, 0] [1, 273, 40, 40]
// CHECK-SAME:  : tensor<1x546x40x40xf16> to tensor<1x273x40x40xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC
// CHECK-SAME:  } : tensor<1x273x40x40xf16> -> tensor<1x40x40x273xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 273]
// CHECK-SAME:  : tensor<1x40x40x273xf16>, tensor<1x40x40x273xf16> -> tensor<1x40x40x546xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x40x40x546xf16>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3, d4, d5, d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d2, d4, d0, d1, d3)>

// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d3, d4, d5, d0)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d2, d4, d0, d1, d3)>

func.func @MemPermuteRank6Split2Part(%arg0: tensor<960x1x12x2x12x2xf16, {order = #map}>) -> tensor<960x2x2x1x12x12xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>, mem_perm = #map1, tilingStrategy = [2, 1, 1, 1, 1, 1]} : tensor<960x1x12x2x12x2xf16, {order = #map}> -> tensor<960x2x2x1x12x12xf16>
  return %0 : tensor<960x2x2x1x12x12xf16>
}
// CHECK-LABEL: @MemPermuteRank6Split2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<960x1x12x2x12x2xf16, {order = #map}>) -> tensor<960x2x2x1x12x12xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0, 0, 0] [480, 1, 12, 2, 12, 2]
// CHECK-SAME:  : tensor<960x1x12x2x12x2xf16, {order = #map}> to tensor<480x1x12x2x12x2xf16, {order = #map}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #map1, mem_perm = #map2
// CHECK-SAME:  } : tensor<480x1x12x2x12x2xf16, {order = #map}> -> tensor<480x2x2x1x12x12xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [480, 0, 0, 0, 0, 0] [480, 1, 12, 2, 12, 2]
// CHECK-SAME:  : tensor<960x1x12x2x12x2xf16, {order = #map}> to tensor<480x1x12x2x12x2xf16, {order = #map}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #map1, mem_perm = #map2
// CHECK-SAME:  } : tensor<480x1x12x2x12x2xf16, {order = #map}> -> tensor<480x2x2x1x12x12xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0, 0, 0], [480, 0, 0, 0, 0, 0]
// CHECK-SAME:  : tensor<480x2x2x1x12x12xf16>, tensor<480x2x2x1x12x12xf16> -> tensor<960x2x2x1x12x12xf16>

// CHECK:       return [[OUTPUT]] : tensor<960x2x2x1x12x12xf16>

// -----

func.func @AvgPoolSwSplit2Part(%arg0: tensor<1x32x1800x16xf16>) -> tensor<1x32x1789x16xf16> {
  %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x1800x16xf16> -> tensor<1x32x1789x16xf16>
  return %0 : tensor<1x32x1789x16xf16>
}
// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1800x16xf16>) -> tensor<1x32x1789x16xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 906, 16]
// CHECK-SAME:  :  tensor<1x32x1800x16xf16> to tensor<1x32x906x16xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
// CHECK-SAME:  } : tensor<1x32x906x16xf16> -> tensor<1x32x895x16xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 895, 0] [1, 32, 905, 16]
// CHECK-SAME:  : tensor<1x32x1800x16xf16> to tensor<1x32x905x16xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
// CHECK-SAME:  } : tensor<1x32x905x16xf16> -> tensor<1x32x894x16xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 895, 0]
// CHECK-SAME:  : tensor<1x32x895x16xf16>, tensor<1x32x894x16xf16> -> tensor<1x32x1789x16xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x32x1789x16xf16>

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
    } -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_TILE:%.+]] =  const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitSparseQuantNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType, {order = #NHWC}>
func.func @SplitSparseQuantNCEConvOverOH(%arg0: tensor<1x32x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<320x1x1x4xsi32>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType, {order = #NHWC}> to tensor<1x32x41x80x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_0]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80x!qElemType, {order = #NHWC}> to tensor<1x32x41x80x!qElemType, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x320x40x80x!qElemType1, {order = #NHWC}>

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x23040xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x23040xf16, {order = #NHWC}>) -> tensor<1x16x1x23040xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x16x1x23040xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x23040xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 7680] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 15360] [1, 16, 7, 7680]
    // CHECK-SAME:      tensor<1x16x7x23040xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x7680xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 7680], [0, 0, 0, 15360]
    // CHECK-SAME:      -> tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x23040xf16, {order = #NHWC}>
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

// CHECK-LABEL: @GRUSequenceForward
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @GRUSequenceForward(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

      // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4xf16>
      // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
      // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x3x10xf16>

      // Tile 0

      // CHECK:     [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE0:%.+]], [[OUTPUTHO_TILE0:%.+]] = VPU.GRUSequence([[INPUT0_TILE0]], [[INPUT1_TILE0]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 1

      // CHECK:     [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT0]] [0, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE1:%.+]], [[OUTPUTHO_TILE1:%.+]] = VPU.GRUSequence([[INPUT0_TILE1]], [[INPUT1_TILE1]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 2

      // CHECK:     [[INPUT0_TILE2:%.+]] = VPU.Slice [[INPUT0]] [1, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE2:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE2:%.+]], [[OUTPUTHO_TILE2:%.+]] = VPU.GRUSequence([[INPUT0_TILE2]], [[INPUT1_TILE2]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 3

      // CHECK:     [[INPUT0_TILE3:%.+]] = VPU.Slice [[INPUT0]] [1, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE3:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE3:%.+]], [[OUTPUTHO_TILE3:%.+]] = VPU.GRUSequence([[INPUT0_TILE3]], [[INPUT1_TILE3]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // CHECK:     [[OUTPUTY:%.+]] = VPU.Concat([[OUTPUTY_TILE0]], [[OUTPUTY_TILE1]], [[OUTPUTY_TILE2]], [[OUTPUTY_TILE3]])
      // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 50000, 0], [1, 0, 0, 0], [1, 0, 50000, 0]]} : tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16> -> tensor<2x1x100000x1xf16>
      // CHECK:     [[OUTPUTHO:%.+]] = VPU.Concat([[OUTPUTHO_TILE0]], [[OUTPUTHO_TILE1]], [[OUTPUTHO_TILE2]], [[OUTPUTHO_TILE3]])
      // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]} : tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<2x1x1xf16>

      // CHECK:     return [[OUTPUTY]], [[OUTPUTHO]] : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL: @GRUSequenceReverse
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @GRUSequenceReverse(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

      // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4xf16>
      // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
      // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x3x10xf16>

      // Tile 0

      // CHECK:     [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE0:%.+]], [[OUTPUTHO_TILE0:%.+]] = VPU.GRUSequence([[INPUT0_TILE0]], [[INPUT1_TILE0]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 1

      // CHECK:     [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT0]] [0, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE1:%.+]], [[OUTPUTHO_TILE1:%.+]] = VPU.GRUSequence([[INPUT0_TILE1]], [[INPUT1_TILE1]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 2

      // CHECK:     [[INPUT0_TILE2:%.+]] = VPU.Slice [[INPUT0]] [1, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE2:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE2:%.+]], [[OUTPUTHO_TILE2:%.+]] = VPU.GRUSequence([[INPUT0_TILE2]], [[INPUT1_TILE2]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 3

      // CHECK:     [[INPUT0_TILE3:%.+]] = VPU.Slice [[INPUT0]] [1, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE3:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x1x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE3:%.+]], [[OUTPUTHO_TILE3:%.+]] = VPU.GRUSequence([[INPUT0_TILE3]], [[INPUT1_TILE3]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // CHECK:     [[OUTPUTY:%.+]] = VPU.Concat([[OUTPUTY_TILE0]], [[OUTPUTY_TILE1]], [[OUTPUTY_TILE2]], [[OUTPUTY_TILE3]])
      // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0, 0], [0, 0, 50000, 0], [1, 0, 0, 0], [1, 0, 50000, 0]]} : tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16> -> tensor<2x1x100000x1xf16>
      // CHECK:     [[OUTPUTHO:%.+]] = VPU.Concat([[OUTPUTHO_TILE0]], [[OUTPUTHO_TILE1]], [[OUTPUTHO_TILE2]], [[OUTPUTHO_TILE3]])
      // CHECK-SAME{LITERAL}:    {static_offsets = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]} : tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<2x1x1xf16>

      // CHECK:     return [[OUTPUTY]], [[OUTPUTHO]] : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL: @GRUSequenceBidirectional
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x2x1xf16>
func.func @GRUSequenceBidirectional(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x2x1xf16>) -> (tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>) {
      %cst = const.Declare tensor<2x3x10xf16> = dense<1.000000e+00> : tensor<2x3x10xf16>
      %cst_0 = const.Declare tensor<2x3x1xf16> = dense<1.000000e+00> : tensor<2x3x1xf16>
      %cst_1 = const.Declare tensor<2x4xf16> = dense<1.000000e+00> : tensor<2x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x2x1xf16>, tensor<2x3x10xf16>, tensor<2x3x1xf16>, tensor<2x4xf16> -> tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>

      // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x4xf16>, [#const.SubView<[1, 0], [1, 4]>]

      // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x1xf16>, [#const.SubView<[1, 0, 0], [1, 3, 1]>]

      // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x3x10xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x10xf16>, [#const.SubView<[1, 0, 0], [1, 3, 10]>

      // CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1x4xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x4xf16>, [#const.SubView<[0, 0], [1, 4]>]

      // CHECK-DAG:       [[CST3:%.+]] = const.Declare tensor<1x3x1xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x1xf16>, [#const.SubView<[0, 0, 0], [1, 3, 1]>]

      // CHECK-DAG:       [[CST4:%.+]] = const.Declare tensor<1x3x10xf16>
      // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x10xf16>, [#const.SubView<[0, 0, 0], [1, 3, 10]>

      // Tile 0

      // CHECK:     [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE0:%.+]], [[OUTPUTHO_TILE0:%.+]] = VPU.GRUSequence([[INPUT0_TILE0]], [[INPUT1_TILE0]], [[CST4]], [[CST3]], [[CST2]])
      // CHECK-SAME:    {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 1

      // CHECK:     [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT0]] [0, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE1:%.+]], [[OUTPUTHO_TILE1:%.+]] = VPU.GRUSequence([[INPUT0_TILE1]], [[INPUT1_TILE1]], [[CST4]], [[CST3]], [[CST2]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 2

      // CHECK:     [[INPUT0_TILE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE2:%.+]] = VPU.Slice [[INPUT1]] [0, 1, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE2:%.+]], [[OUTPUTHO_TILE2:%.+]] = VPU.GRUSequence([[INPUT0_TILE2]], [[INPUT1_TILE2]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 3

      // CHECK:     [[INPUT0_TILE3:%.+]] = VPU.Slice [[INPUT0]] [0, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE3:%.+]] = VPU.Slice [[INPUT1]] [0, 1, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE3:%.+]], [[OUTPUTHO_TILE3:%.+]] = VPU.GRUSequence([[INPUT0_TILE3]], [[INPUT1_TILE3]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 4

      // CHECK:     [[INPUT0_TILE4:%.+]] = VPU.Slice [[INPUT0]] [1, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE4:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE4:%.+]], [[OUTPUTHO_TILE4:%.+]] = VPU.GRUSequence([[INPUT0_TILE4]], [[INPUT1_TILE4]], [[CST4]], [[CST3]], [[CST2]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 5

      // CHECK:     [[INPUT0_TILE5:%.+]] = VPU.Slice [[INPUT0]] [1, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE5:%.+]] = VPU.Slice [[INPUT1]] [1, 0, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE5:%.+]], [[OUTPUTHO_TILE5:%.+]] = VPU.GRUSequence([[INPUT0_TILE5]], [[INPUT1_TILE5]], [[CST4]], [[CST3]], [[CST2]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 6

      // CHECK:     [[INPUT0_TILE6:%.+]] = VPU.Slice [[INPUT0]] [1, 0, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE6:%.+]] = VPU.Slice [[INPUT1]] [1, 1, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE6:%.+]], [[OUTPUTHO_TILE6:%.+]] = VPU.GRUSequence([[INPUT0_TILE6]], [[INPUT1_TILE6]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // Tile 7

      // CHECK:     [[INPUT0_TILE7:%.+]] = VPU.Slice [[INPUT0]] [1, 50000, 0] [1, 50000, 10] : tensor<2x100000x10xf16> to tensor<1x50000x10xf16>
      // CHECK:     [[INPUT1_TILE7:%.+]] = VPU.Slice [[INPUT1]] [1, 1, 0] [1, 1, 1] : tensor<2x2x1xf16> to tensor<1x1x1xf16>
      // CHECK:     [[OUTPUTY_TILE7:%.+]], [[OUTPUTHO_TILE7:%.+]] = VPU.GRUSequence([[INPUT0_TILE7]], [[INPUT1_TILE7]], [[CST1]], [[CST0]], [[CST]])
      // CHECK-SAME:{clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 50000 : i64, should_linear_before_reset} : tensor<1x50000x10xf16>, tensor<1x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<1x1x50000x1xf16>, tensor<1x1x1xf16>

      // CHECK:     [[OUTPUTY:%.+]] = VPU.Concat([[OUTPUTY_TILE0]], [[OUTPUTY_TILE1]], [[OUTPUTY_TILE2]], [[OUTPUTY_TILE3]], [[OUTPUTY_TILE4]], [[OUTPUTY_TILE5]], [[OUTPUTY_TILE6]], [[OUTPUTY_TILE7]])
      // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 0, 50000, 0], [0, 1, 0, 0], [0, 1, 50000, 0], [1, 0, 0, 0], [1, 0, 50000, 0], [1, 1, 0, 0], [1, 1, 50000, 0]]}
      // CHECK-SAME:            tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16>, tensor<1x1x50000x1xf16> -> tensor<2x2x100000x1xf16>
      // CHECK:     [[OUTPUTHO:%.+]] = VPU.Concat([[OUTPUTHO_TILE0]], [[OUTPUTHO_TILE1]], [[OUTPUTHO_TILE2]], [[OUTPUTHO_TILE3]], [[OUTPUTHO_TILE4]], [[OUTPUTHO_TILE5]], [[OUTPUTHO_TILE6]], [[OUTPUTHO_TILE7]])
      // CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]]}
      // CHECK-SAME:            tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<2x2x1xf16>
      // CHECK:     return [[OUTPUTY]], [[OUTPUTHO]] : tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
}


// -----

// CHECK-LABEL:   func.func @GridSampleSplit
func.func @GridSampleSplit(%arg0: tensor<1x3x272x480xf16>, %arg1: tensor<1x272x480x2xf16>) -> tensor<1x3x272x480xf16> {
      %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>
      return %0 : tensor<1x3x272x480xf16>

      // Tile 0

      // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 2, 272, 480] : tensor<1x3x272x480xf16> to tensor<1x2x272x480xf16>

      // CHECK:     %1 = VPU.GridSample(%0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x2x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x2x272x480xf16>

      // Tile 1

      // CHECK:     %2 = VPU.Slice %arg0 [0, 2, 0, 0] [1, 1, 272, 480] : tensor<1x3x272x480xf16> to tensor<1x1x272x480xf16>

      // CHECK:     %3 = VPU.GridSample(%2, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x1x272x480xf16>

      // CHECK:     %4 = VPU.Concat(%1, %3)
      // CHECK-SAME:[0, 0, 0, 0], [0, 2, 0, 0]
      // CHECK-SAME:-> tensor<1x3x272x480xf16>

      // CHECK:     return %4 : tensor<1x3x272x480xf16>

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
        strides = [1, 1]
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpNearestTileOverH
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x64x64x48xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<64x64x1x1xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<64x1x1x4xsi32>
func.func @NCEInterpNearestTileOverH(
                %data: tensor<1x64x64x48xf16, {order = #NHWC}>,
                %weights: tensor<64x64x1x1xf16, {order = #NHWC}>,
                %weightsTable: tensor<64x1x1x4xsi32>)
          -> tensor<1x64x128x96xf16, {order = #NHWC}> {
    %sparsity_map = const.Declare tensor<1x64x128x96xi1> = dense<1> : tensor<1x64x128x96xi1>
    %se_table = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 64, 48],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 128, 96]>
    } -> tensor<1x1x128x96xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsity_map, %se_table) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 128, 96]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x64x48xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x128x96xi1>,
            storage_element_table=tensor<1x1x128x96xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 128, 96]>>

    %interp = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x128x96xf16, {order = #NHWC}>

    return  %interp : tensor<1x64x128x96xf16, {order = #NHWC}>

    // Tiled SMs
    // CHECK-DAG:           [[SM_TILE_1:%.+]] = const.Declare tensor<1x64x64x96xi1> = dense<true> : tensor<1x64x128x96xi1>, [#const.SubView<[0, 0, 64, 0], [1, 64, 64, 96]>]
    // CHECK-DAG:           [[SM_TILE_0:%.+]] = const.Declare tensor<1x64x64x96xi1> = dense<true> : tensor<1x64x128x96xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 64, 96]>]

    // Tiled StorageElementTable ops
    // CHECK:               [[SET_TILE_0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:              {dataElemType = i32,
    // CHECK-SAME:              dataShape = [1, 64, 32, 48],
    // CHECK-SAME:              seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                  nearest_mode = <FLOOR>,
    // CHECK-SAME:                  offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>,
    // CHECK-SAME:              seDepth = 1 : i64,
    // CHECK-SAME:              seSize = 64 : i64}
    // CHECK-SAME:          -> tensor<1x1x64x96xi32, {order = #NHWC}>
    // CHECK:               [[SET_TILE_1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:              {dataElemType = i32,
    // CHECK-SAME:              dataShape = [1, 64, 32, 48],
    // CHECK-SAME:              seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                  nearest_mode = <FLOOR>,
    // CHECK-SAME:                  offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>,
    // CHECK-SAME:              seDepth = 1 : i64,
    // CHECK-SAME:              seSize = 64 : i64}
    // CHECK-SAME:          -> tensor<1x1x64x96xi32, {order = #NHWC}>

    // TILE 1
    // CHECK:               [[DATA_TILE_1:%.+]] = VPU.Slice [[DATA]] [0, 0, 32, 0] [1, 64, 32, 48]
    // CHECK-SAME:                                tensor<1x64x64x48xf16, {order = #NHWC}> to tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:               [[SPARSE_DATA_TILE_1:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_1]], [[SM_TILE_1]], [[SET_TILE_1]])
    // CHECK-SAME:              {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                  nearest_mode = <FLOOR>,
    // CHECK-SAME:                  offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>}
    // CHECK-SAME:              -> !VPU.SparseTensor<data=tensor<1x64x32x48xf16, {order = #NHWC}>,
    // CHECK-SAME:              sparsity_map=tensor<1x64x64x96xi1>,
    // CHECK-SAME:              storage_element_table=tensor<1x1x64x96xi32, {order = #NHWC}>,
    // CHECK-SAME:              #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                  nearest_mode = <FLOOR>,
    // CHECK-SAME:                  offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>>

    // TILE 0
    // CHECK:               [[DATA_TILE_0:%.+]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 32, 48]
    // CHECK-SAME:                                tensor<1x64x64x48xf16, {order = #NHWC}> to tensor<1x64x32x48xf16, {order = #NHWC}>

    // CHECK:               [[SPARSE_DATA_TILE_0:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_0]], [[SM_TILE_0]], [[SET_TILE_0]])
    // CHECK-SAME:              {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                  nearest_mode = <FLOOR>,
    // CHECK-SAME:                  offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>}
    // CHECK-SAME:              -> !VPU.SparseTensor<data=tensor<1x64x32x48xf16, {order = #NHWC}>,
    // CHECK-SAME:                  sparsity_map=tensor<1x64x64x96xi1>,
    // CHECK-SAME:                  storage_element_table=tensor<1x1x64x96xi32, {order = #NHWC}>,
    // CHECK-SAME:                  #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                      nearest_mode = <FLOOR>,
    // CHECK-SAME:                      offsets = [0, 0, 0, 0], sizes = [1, 64, 64, 96]>>

    // NCE Interpolate ops
    // CHECK:               [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:              {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:              rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:              scales_attr = [2, 2],
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> tensor<1x64x64x96xf16, {order = #NHWC}>
    // CHECK:               [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_1]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:              {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:              rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:              scales_attr = [2, 2],
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> tensor<1x64x64x96xf16, {order = #NHWC}>

    // Concat results
    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]}
    // CHECK-SAME:              : tensor<1x64x64x96xf16, {order = #NHWC}>, tensor<1x64x64x96xf16, {order = #NHWC}> -> tensor<1x64x128x96xf16, {order = #NHWC}>

    // CHECK:               return [[CONCAT]] : tensor<1x64x128x96xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpTileOverC
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x128x48x48xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<128x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<128x1x1x4xsi32>
func.func @NCEInterpTileOverC(
                %data: tensor<1x128x48x48xf16, {order = #NHWC}>,
                %weights: tensor<128x128x1x1xf16, {order = #NHWC}>,
                %weightsTable: tensor<128x1x1x4xsi32>)
          -> tensor<1x128x96x96xf16, {order = #NHWC}> {
    %sparsity_map = const.Declare tensor<1x128x96x96xi1> = dense<1> : tensor<1x128x96x96xi1>
    %se_table = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 128,
        dataShape = [1, 128, 48, 48],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 96, 96]>
    } -> tensor<1x1x96x96xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsity_map, %se_table) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 96, 96]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x128x48x48xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x128x96x96xi1>,
            storage_element_table=tensor<1x1x96x96xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 128, 96, 96]>>

    %interp = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [128, 128, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x96x96xf16, {order = #NHWC}>

    return  %interp : tensor<1x128x96x96xf16, {order = #NHWC}>

    // Tile over C implies full input data for each tile
    // CHECK:       [[SM:%.+]] = const.Declare tensor<1x128x96x96xi1> = dense<true> : tensor<1x128x96x96xi1>
    // CHECK:       [[SE_TABLE:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:    dataElemType = i32,
    // CHECK-SAME:    dataShape = [1, 128, 48, 48],
    // CHECK-SAME:    seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:             nearest_mode = <FLOOR>,
    // CHECK-SAME:             offsets = [0, 0, 0, 0], sizes = [1, 128, 96, 96]>,
    // CHECK-SAME:    seDepth = 1 : i64,
    // CHECK-SAME:    seSize = 128 : i64}
    // CHECK-SAME:    -> tensor<1x1x96x96xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA:%.+]] = VPU.GroupSparseTensor([[DATA]], [[SM]], [[SE_TABLE]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 128, 96, 96]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x128x48x48xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x128x96x96xi1>,
    // CHECK-SAME:                           storage_element_table=tensor<1x1x96x96xi32, {order = #NHWC}>,
    // CHECK-SAME:          #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:              nearest_mode = <FLOOR>,
    // CHECK-SAME:              offsets = [0, 0, 0, 0], sizes = [1, 128, 96, 96]>>

    // Tile 0 weights and Interpolate op
    // CHECK:       [[WEIGHTS_TILE_0:%.+]] = VPU.Slice [[WEIGHTS]] [0, 0, 0, 0] [48, 128, 1, 1] : tensor<128x128x1x1xf16, {order = #NHWC}> to tensor<48x128x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WT_TILE_0:%.+]] = VPU.Slice [[WT]] [0, 0, 0, 0] [48, 1, 1, 4] : tensor<128x1x1x4xsi32> to tensor<48x1x1x4xsi32>

    // CHECK:       [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA]], [[WEIGHTS_TILE_0]], [[WT_TILE_0]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [48, 128, 1, 1],
    // CHECK-SAME:       scales_attr = [2, 2],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:       -> tensor<1x48x96x96xf16, {order = #NHWC}>

    // Tile 1 weights and Interpolate op
    // CHECK:       [[WEIGHTS_TILE_1:%.+]] = VPU.Slice [[WEIGHTS]] [48, 0, 0, 0] [48, 128, 1, 1] : tensor<128x128x1x1xf16, {order = #NHWC}> to tensor<48x128x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WT_TILE_1:%.+]] = VPU.Slice [[WT]] [48, 0, 0, 0] [48, 1, 1, 4] : tensor<128x1x1x4xsi32> to tensor<48x1x1x4xsi32>

    // CHECK:       [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA]], [[WEIGHTS_TILE_1]], [[WT_TILE_1]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [48, 128, 1, 1],
    // CHECK-SAME:       scales_attr = [2, 2],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x48x96x96xf16, {order = #NHWC}>

    // Tile 2 weights and Interpolate op
    // CHECK:       [[WEIGHTS_TILE_2:%.+]] = VPU.Slice [[WEIGHTS]] [96, 0, 0, 0] [32, 128, 1, 1] : tensor<128x128x1x1xf16, {order = #NHWC}> to tensor<32x128x1x1xf16, {order = #NHWC}>
    // CHECK:       [[WT_TILE_2:%.+]] = VPU.Slice [[WT]] [96, 0, 0, 0] [32, 1, 1, 4] : tensor<128x1x1x4xsi32> to tensor<32x1x1x4xsi32>

    // CHECK:       [[INTERP_TILE_2:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA]], [[WEIGHTS_TILE_2]], [[WT_TILE_2]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:       rawFilterShape = [32, 128, 1, 1],
    // CHECK-SAME:       scales_attr = [2, 2],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x32x96x96xf16, {order = #NHWC}>

    // Concatenate all tiles
    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]], [[INTERP_TILE_2]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0], [0, 96, 0, 0]]}
    // CHECK-SAME:          tensor<1x48x96x96xf16, {order = #NHWC}>, tensor<1x48x96x96xf16, {order = #NHWC}>, tensor<1x32x96x96xf16, {order = #NHWC}> -> tensor<1x128x96x96xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x128x96x96xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpBilinearTileOverH
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x64x48x48xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<64x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<64x1x1x4xsi32>
func.func @NCEInterpBilinearTileOverH(
                %data: tensor<1x64x48x48xf16, {order = #NHWC}>,
                %weights: tensor<64x64x3x3xf16, {order = #NHWC}>,
                %weightsTable: tensor<64x1x1x4xsi32>)
          -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    %sparsityMap = const.Declare tensor<1x64x144x144xi1> = dense<1> : tensor<1x64x144x144xi1>
    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 48, 48],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 3.0, 3.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 144, 144]>
    } -> tensor<1x1x144x144xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 3.0, 3.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 144, 144]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x48x48xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x144x144xi1>,
            storage_element_table=tensor<1x1x144x144xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 3.0, 3.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 144, 144]>>

    %task = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [3, 3],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>

    return  %task : tensor<1x64x144x144xf16, {order = #NHWC}>

    // Tiled Sparsity Map constants for input data
    // CHECK-DAG:   [[SM_TILE_1:%.+]] = const.Declare tensor<1x64x72x144xi1> = dense<true> : tensor<1x64x144x144xi1>, [#const.SubView<[0, 0, 72, 0], [1, 64, 72, 144]>]
    // CHECK-DAG:   [[SM_TILE_0:%.+]] = const.Declare tensor<1x64x72x144xi1> = dense<true> : tensor<1x64x144x144xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 72, 144]>]

    // Tiled Storage Element Table operations
    // CHECK:       [[SET_TILE_0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 24, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x72x144xi32, {order = #NHWC}>

    // CHECK:       [[SET_TILE_1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 24, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x72x144xi32, {order = #NHWC}>

    // Tile 1 data
    // CHECK:       [[DATA_TILE_1:%.+]] = VPU.Slice [[DATA]] [0, 0, 24, 0] [1, 64, 24, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x24x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_1:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_1]], [[SM_TILE_1]], [[SET_TILE_1]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x24x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x72x144xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x72x144xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>>

    // Tile 0 data
    // CHECK:       [[DATA_TILE_0:%.+]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 24, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x24x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_0:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_0]], [[SM_TILE_0]], [[SET_TILE_0]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x24x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x72x144xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x72x144xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 0, 0], sizes = [1, 64, 72, 144]>>

    // Interpolate operations for tile 0 and 1
    // CHECK:       [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:       scales_attr = [3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x72x144xf16, {order = #NHWC}>
    // CHECK:       [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_1]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 1, 1],
    // CHECK-SAME:       scales_attr = [3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x72x144xf16, {order = #NHWC}>

    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 72, 0]]}
    // CHECK-SAME:          tensor<1x64x72x144xf16, {order = #NHWC}>, tensor<1x64x72x144xf16, {order = #NHWC}> -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x64x144x144xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpBilinearPytorchHalfPixelOddScalesTileOverH
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x64x48x48xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<64x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<64x1x1x4xsi32>
func.func @NCEInterpBilinearPytorchHalfPixelOddScalesTileOverH(
                %data: tensor<1x64x48x48xf16, {order = #NHWC}>,
                %weights: tensor<64x64x3x3xf16, {order = #NHWC}>,
                %weightsTable: tensor<64x1x1x4xsi32>)
          -> tensor<1x64x144x144xf16, {order = #NHWC}> {
    %sparsityMap = const.Declare tensor<1x64x146x146xi1> = dense<1> : tensor<1x64x146x146xi1>
    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 48, 48],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 3.0, 3.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 146, 146]>
    } -> tensor<1x1x146x146xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
            scale = [1.0, 1.0, 3.0, 3.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 146, 146]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x48x48xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x146x146xi1>,
            storage_element_table=tensor<1x1x146x146xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>,
                scale = [1.0, 1.0, 3.0, 3.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 146, 146]>>

    %task = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [3, 3],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x144x144xf16, {order = #NHWC}>

    return  %task : tensor<1x64x144x144xf16, {order = #NHWC}>

    // Tiled Sparsity Map constants for input data
    // CHECK-DAG:   [[SM_TILE_1:%.+]] = const.Declare tensor<1x64x74x146xi1> = dense<true> : tensor<1x64x146x146xi1>, [#const.SubView<[0, 0, 72, 0], [1, 64, 74, 146]>]
    // CHECK-DAG:   [[SM_TILE_0:%.+]] = const.Declare tensor<1x64x74x146xi1> = dense<true> : tensor<1x64x146x146xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 74, 146]>]

    // Tiled Storage Element Table operations
    // CHECK:       [[SET_TILE_0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 25, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 74, 146]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x74x146xi32, {order = #NHWC}>

    // CHECK:       [[SET_TILE_1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 25, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 3, 0], sizes = [1, 64, 74, 146]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x74x146xi32, {order = #NHWC}>

    // Tile 1 data
    // CHECK:       [[DATA_TILE_1:%.+]] = VPU.Slice [[DATA]] [0, 0, 23, 0] [1, 64, 25, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x25x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_1:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_1]], [[SM_TILE_1]], [[SET_TILE_1]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 3, 0], sizes = [1, 64, 74, 146]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x25x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x74x146xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x74x146xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 3, 0], sizes = [1, 64, 74, 146]>>

    // Tile 0 data
    // CHECK:       [[DATA_TILE_0:%.+]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 25, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x25x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_0:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_0]], [[SM_TILE_0]], [[SET_TILE_0]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 74, 146]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x25x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x74x146xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x74x146xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <PYTORCH_HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 0, 0], sizes = [1, 64, 74, 146]>>

    // Interpolate operations for tile 0 and 1
    // CHECK:       [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       scales_attr = [3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x72x144xf16, {order = #NHWC}>
    // CHECK:       [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_1]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       scales_attr = [3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x72x144xf16, {order = #NHWC}>

    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 72, 0]]}
    // CHECK-SAME:          tensor<1x64x72x144xf16, {order = #NHWC}>, tensor<1x64x72x144xf16, {order = #NHWC}> -> tensor<1x64x144x144xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x64x144x144xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpBilinearHalfPixelEvenScalesTileOverH
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x64x60x60xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<64x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<64x1x1x4xsi32>
func.func @NCEInterpBilinearHalfPixelEvenScalesTileOverH(
                %data: tensor<1x64x60x60xf16, {order = #NHWC}>,
                %weights: tensor<64x64x3x3xf16, {order = #NHWC}>,
                %weightsTable: tensor<64x1x1x4xsi32>)
          -> tensor<1x64x120x120xf16, {order = #NHWC}> {
    %sparsityMap = const.Declare tensor<1x64x122x122xi1> = dense<1> : tensor<1x64x122x122xi1>
    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 60, 60],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 122, 122]>
    } -> tensor<1x1x122x122xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <HALF_PIXEL>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 122, 122]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x60x60xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x122x122xi1>,
            storage_element_table=tensor<1x1x122x122xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <HALF_PIXEL>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 122, 122]>>

    %task = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x120x120xf16, {order = #NHWC}>

    return  %task : tensor<1x64x120x120xf16, {order = #NHWC}>

    // Tiled Sparsity Map constants for input data
    // CHECK-DAG:   [[SM_TILE_1:%.+]] = const.Declare tensor<1x64x62x122xi1> = dense<true> : tensor<1x64x122x122xi1>, [#const.SubView<[0, 0, 60, 0], [1, 64, 62, 122]>]
    // CHECK-DAG:   [[SM_TILE_0:%.+]] = const.Declare tensor<1x64x62x122xi1> = dense<true> : tensor<1x64x122x122xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 62, 122]>]

    // Tiled Storage Element Table operations
    // CHECK:       [[SET_TILE_0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 31, 60],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 62, 122]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x62x122xi32, {order = #NHWC}>

    // CHECK:       [[SET_TILE_1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 31, 60],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 2, 0], sizes = [1, 64, 62, 122]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x62x122xi32, {order = #NHWC}>

    // Tile 1 data
    // CHECK:       [[DATA_TILE_1:%.+]] = VPU.Slice [[DATA]] [0, 0, 29, 0] [1, 64, 31, 60]
    // CHECK-SAME:      tensor<1x64x60x60xf16, {order = #NHWC}> to tensor<1x64x31x60xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_1:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_1]], [[SM_TILE_1]], [[SET_TILE_1]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 2, 0], sizes = [1, 64, 62, 122]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x31x60xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x62x122xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x62x122xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 2, 0], sizes = [1, 64, 62, 122]>>

    // Tile 0 data
    // CHECK:       [[DATA_TILE_0:%.+]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 31, 60]
    // CHECK-SAME:      tensor<1x64x60x60xf16, {order = #NHWC}> to tensor<1x64x31x60xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_0:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_0]], [[SM_TILE_0]], [[SET_TILE_0]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 62, 122]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x31x60xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x62x122xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x62x122xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:      offsets = [0, 0, 0, 0], sizes = [1, 64, 62, 122]>>

    // Interpolate operations for tile 0 and 1
    // CHECK:       [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       scales_attr = [2, 2],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x60x120xf16, {order = #NHWC}>
    // CHECK:       [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_1]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       scales_attr = [2, 2],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x60x120xf16, {order = #NHWC}>

    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 60, 0]]}
    // CHECK-SAME:          tensor<1x64x60x120xf16, {order = #NHWC}>, tensor<1x64x60x120xf16, {order = #NHWC}> -> tensor<1x64x120x120xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x64x120x120xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NCEInterpBilinearAlignCornersTileOverH
// CHECK-SAME:        [[DATA:%arg[0-9]]]: tensor<1x64x48x48xf16, {order = #NHWC}>
// CHECK-SAME:        [[WEIGHTS:%arg[0-9]]]: tensor<64x64x3x3xf16, {order = #NHWC}>
// CHECK-SAME:        [[WT:%arg[0-9]]]: tensor<64x1x1x4xsi32>
func.func @NCEInterpBilinearAlignCornersTileOverH(
                %data: tensor<1x64x48x48xf16, {order = #NHWC}>,
                %weights: tensor<64x64x3x3xf16, {order = #NHWC}>,
                %weightsTable: tensor<64x1x1x4xsi32>)
          -> tensor<1x64x142x142xf16, {order = #NHWC}> {
    %sparsityMap = const.Declare tensor<1x64x144x144xi1> = dense<1> : tensor<1x64x144x144xi1>
    %storageElement = VPU.StorageElementTable {
        dataElemType = i32,
        seDepth = 1,
        seSize = 64,
        dataShape = [1, 64, 48, 48],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ALIGN_CORNERS>,
            scale = [1.0, 1.0, 3.0, 3.0],
            initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>
    } -> tensor<1x1x144x144xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%data, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ALIGN_CORNERS>,
            scale = [1.0, 1.0, 3.0, 3.0],
            initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x48x48xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x144x144xi1>,
            storage_element_table=tensor<1x1x144x144xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ALIGN_CORNERS>,
                scale = [1.0, 1.0, 3.0, 3.0],
                initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>>

    %task = VPU.NCE.Interpolate(%input, %weights, %weightsTable) {
        rawFilterShape = [64, 64, 3, 3],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x142x142xf16, {order = #NHWC}>

    return  %task : tensor<1x64x142x142xf16, {order = #NHWC}>

    // Tiled Sparsity Map constants for input data
    // CHECK-DAG:   [[SM_TILE_1:%.+]] = const.Declare tensor<1x64x73x144xi1> = dense<true> : tensor<1x64x144x144xi1>, [#const.SubView<[0, 0, 71, 0], [1, 64, 73, 144]>]
    // CHECK-DAG:   [[SM_TILE_0:%.+]] = const.Declare tensor<1x64x73x144xi1> = dense<true> : tensor<1x64x144x144xi1>, [#const.SubView<[0, 0, 0, 0], [1, 64, 73, 144]>]

    // Tiled Storage Element Table operations
    // CHECK:       [[SET_TILE_0:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 25, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x73x144xi32, {order = #NHWC}>

    // CHECK:       [[SET_TILE_1:%.+]] = VPU.StorageElementTable
    // CHECK-SAME:      {dataElemType = i32,
    // CHECK-SAME:       dataShape = [1, 64, 25, 48],
    // CHECK-SAME:       seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 2, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>,
    // CHECK-SAME:       seDepth = 1 : i64,
    // CHECK-SAME:       seSize = 64 : i64}
    // CHECK-SAME:      -> tensor<1x1x73x144xi32, {order = #NHWC}>

    // Tile 1 data
    // CHECK:       [[DATA_TILE_1:%.+]] = VPU.Slice [[DATA]] [0, 0, 23, 0] [1, 64, 25, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x25x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_1:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_1]], [[SM_TILE_1]], [[SET_TILE_1]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 2, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x25x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x73x144xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x73x144xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 2, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>>

    // Tile 0 data
    // CHECK:       [[DATA_TILE_0:%.+]] = VPU.Slice [[DATA]] [0, 0, 0, 0] [1, 64, 25, 48]
    // CHECK-SAME:      tensor<1x64x48x48xf16, {order = #NHWC}> to tensor<1x64x25x48xf16, {order = #NHWC}>

    // CHECK:       [[SPARSE_DATA_TILE_0:%.+]] = VPU.GroupSparseTensor([[DATA_TILE_0]], [[SM_TILE_0]], [[SET_TILE_0]])
    // CHECK-SAME:      {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>}
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x64x25x48xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x73x144xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x73x144xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ALIGN_CORNERS>, scale = [1.000000e+00, 1.000000e+00, 3.000000e+00, 3.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0], sizes = [1, 64, 73, 144], initial_input_shape = [1, 64, 48, 48], initial_output_shape = [1, 64, 142, 142]>>

    // Interpolate operations for tile 0 and 1
    // CHECK:       [[INTERP_TILE_0:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_0]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x71x142xf16, {order = #NHWC}>
    // CHECK:       [[INTERP_TILE_1:%.+]] = VPU.NCE.Interpolate([[SPARSE_DATA_TILE_1]], [[WEIGHTS]], [[WT]])
    // CHECK-SAME:      {mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:       rawFilterShape = [64, 64, 3, 3],
    // CHECK-SAME:       strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x64x71x142xf16, {order = #NHWC}>

    // CHECK:               [[CONCAT:%.+]] = VPU.Concat([[INTERP_TILE_0]], [[INTERP_TILE_1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 0, 71, 0]]}
    // CHECK-SAME:          tensor<1x64x71x142xf16, {order = #NHWC}>, tensor<1x64x71x142xf16, {order = #NHWC}> -> tensor<1x64x142x142xf16, {order = #NHWC}>

    // CHECK:       return [[CONCAT]] : tensor<1x64x142x142xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ReduceSumOneAxisKeepDimsFalse
// CHECK-SAME:    [[INPUT:%arg[0-9]]]: tensor<1x20x175x512xf16>

func.func @ReduceSumOneAxisKeepDimsFalse(%arg0: tensor<1x20x175x512xf16>) -> tensor<1x20x512xf16> {
    %0 = VPU.ReduceSum(%arg0) {axes_value = [2]} : tensor<1x20x175x512xf16> -> tensor<1x20x512xf16>
    return %0 : tensor<1x20x512xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 175, 256] : tensor<1x20x175x512xf16> to tensor<1x20x175x256xf16>
// CHECK:    [[REDUCE_SUM0:%.+]] = VPU.ReduceSum([[SLICE0]]) {axes_value = [2]} : tensor<1x20x175x256xf16> -> tensor<1x20x256xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 256] [1, 20, 175, 256] : tensor<1x20x175x512xf16> to tensor<1x20x175x256xf16>
// CHECK:    [[REDUCE_SUM1:%.+]] = VPU.ReduceSum([[SLICE1]]) {axes_value = [2]} : tensor<1x20x175x256xf16> -> tensor<1x20x256xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[REDUCE_SUM0]], [[REDUCE_SUM1]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0, 0], [0, 0, 256]]} : tensor<1x20x256xf16>, tensor<1x20x256xf16> -> tensor<1x20x512xf16>
// CHECK:    return [[CONCAT]] : tensor<1x20x512xf16>
}

// -----

// CHECK-LABEL: @ReduceSumThreeAxesKeepDimsFalse
// CHECK-SAME:   [[INPUT:%arg[0-9]]]: tensor<1x20x175x512xf16>

func.func @ReduceSumThreeAxesKeepDimsFalse(%arg0: tensor<1x20x175x512xf16>) -> tensor<175xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [0, 1, 3]} : tensor<1x20x175x512xf16> -> tensor<175xf16>
  return %0 : tensor<175xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 88, 512] : tensor<1x20x175x512xf16> to tensor<1x20x88x512xf16>
// CHECK:    [[REDUCE_SUM0:%.+]] = VPU.ReduceSum([[SLICE0]]) {axes_value = [0, 1, 3]} : tensor<1x20x88x512xf16> -> tensor<88xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 88, 0] [1, 20, 87, 512] : tensor<1x20x175x512xf16> to tensor<1x20x87x512xf16>
// CHECK:    [[REDUCE_SUM1:%.+]] = VPU.ReduceSum([[SLICE1]]) {axes_value = [0, 1, 3]} : tensor<1x20x87x512xf16> -> tensor<87xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[REDUCE_SUM0]], [[REDUCE_SUM1]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0], [88]]} : tensor<88xf16>, tensor<87xf16> -> tensor<175xf16>
// CHECK:    return [[CONCAT]] : tensor<175xf16>
}

// -----

// CHECK-LABEL: func.func @ReduceL1OverLargestDimsKeepDimsFalse(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x256x256x20xf16>) -> tensor<1x20xf16> {
func.func @ReduceL1OverLargestDimsKeepDimsFalse(%arg0: tensor<1x256x256x20xf16>) -> tensor<1x20xf16> {
  %0 = VPU.ReduceL1(%arg0) {axes_value = [1, 2]} : tensor<1x256x256x20xf16> -> tensor<1x20xf16>
  return %0 : tensor<1x20xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 256, 256, 10] : tensor<1x256x256x20xf16> to tensor<1x256x256x10xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceL1(%[[VAL_1]]) {axes_value = [1, 2]} : tensor<1x256x256x10xf16> -> tensor<1x10xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 10] [1, 256, 256, 10] : tensor<1x256x256x20xf16> to tensor<1x256x256x10xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceL1(%[[VAL_3]]) {axes_value = [1, 2]} : tensor<1x256x256x10xf16> -> tensor<1x10xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]]) {static_offsets = {{\[\[}}0, 0], [0, 10]]} : tensor<1x10xf16>, tensor<1x10xf16> -> tensor<1x20xf16>
// CHECK:   return %[[VAL_5]] : tensor<1x20xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceL2OverLargestAxesKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x256x256x20xf16>) -> tensor<1x1x1x20xf16> {
func.func @ReduceL2OverLargestAxesKeepDimsTrue(%arg0: tensor<1x256x256x20xf16>) -> tensor<1x1x1x20xf16> {
  %0 = VPU.ReduceL2(%arg0) {axes_value = [1, 2], keep_dims} : tensor<1x256x256x20xf16> -> tensor<1x1x1x20xf16>
  return %0 : tensor<1x1x1x20xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 256, 256, 10] : tensor<1x256x256x20xf16> to tensor<1x256x256x10xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceL2(%[[VAL_1]]) {axes_value = [1, 2], keep_dims} : tensor<1x256x256x10xf16> -> tensor<1x1x1x10xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 10] [1, 256, 256, 10] : tensor<1x256x256x20xf16> to tensor<1x256x256x10xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceL2(%[[VAL_3]]) {axes_value = [1, 2], keep_dims} : tensor<1x256x256x10xf16> -> tensor<1x1x1x10xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 10]]} : tensor<1x1x1x10xf16>, tensor<1x1x1x10xf16> -> tensor<1x1x1x20xf16>
// CHECK:   return %[[VAL_5]] : tensor<1x1x1x20xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceLogicalAndSplitOverNCKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x2x1x1xf16> {
func.func @ReduceLogicalAndSplitOverNCKeepDimsTrue(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x2x1x1xf16> {
  %0 = VPU.ReduceLogicalAnd(%arg0) {axes_value = [2, 3], keep_dims} : tensor<2x2x680x1024xf16> -> tensor<2x2x1x1xf16>
  return %0 : tensor<2x2x1x1xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceLogicalAnd(%[[VAL_1]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x680x1024xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 1, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalAnd(%[[VAL_3]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x680x1024xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [1, 0, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceLogicalAnd(%[[VAL_5]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x680x1024xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [1, 1, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceLogicalAnd(%[[VAL_7]]) {axes_value = [2, 3], keep_dims} : tensor<1x1x680x1024xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<2x2x1x1xf16>
// CHECK:   return %[[VAL_9]] : tensor<2x2x1x1xf16>
// CHECK:   }

}

// -----

// CHECK-LABEL: func.func @ReduceLogicalOrSplitOverHKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x1x680x1xf16> {
func.func @ReduceLogicalOrSplitOverHKeepDimsTrue(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x1x680x1xf16> {
  %0 = VPU.ReduceLogicalOr(%arg0) {axes_value = [1, 3], keep_dims} : tensor<2x2x680x1024xf16> -> tensor<2x1x680x1xf16>
  return %0 : tensor<2x1x680x1xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 227, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x227x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceLogicalOr(%[[VAL_1]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x227x1024xf16> -> tensor<2x1x227x1xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 227, 0] [2, 2, 227, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x227x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalOr(%[[VAL_3]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x227x1024xf16> -> tensor<2x1x227x1xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 454, 0] [2, 2, 226, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x226x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceLogicalOr(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x226x1024xf16> -> tensor<2x1x226x1xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 227, 0], [0, 0, 454, 0]]} : tensor<2x1x227x1xf16>, tensor<2x1x227x1xf16>, tensor<2x1x226x1xf16> -> tensor<2x1x680x1xf16>
// CHECK:   return %[[VAL_7]] : tensor<2x1x680x1xf16>
// CHECK:         }
}

// -----

// CHECK-LABEL: func.func @ReduceMaxSplitOverWKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x1x1x1024xf16> {
func.func @ReduceMaxSplitOverWKeepDimsTrue(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x1x1x1024xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x1024xf16> -> tensor<2x1x1x1024xf16>
  return %0 : tensor<2x1x1x1024xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 680, 342] : tensor<2x2x680x1024xf16> to tensor<2x2x680x342xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceMax(%[[VAL_1]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x342xf16> -> tensor<2x1x1x342xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 342] [2, 2, 680, 341] : tensor<2x2x680x1024xf16> to tensor<2x2x680x341xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceMax(%[[VAL_3]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x341xf16> -> tensor<2x1x1x341xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 683] [2, 2, 680, 341] : tensor<2x2x680x1024xf16> to tensor<2x2x680x341xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceMax(%[[VAL_5]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x341xf16> -> tensor<2x1x1x341xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 342], [0, 0, 0, 683]]} : tensor<2x1x1x342xf16>, tensor<2x1x1x341xf16>, tensor<2x1x1x341xf16> -> tensor<2x1x1x1024xf16>
// CHECK:   return %[[VAL_7]] : tensor<2x1x1x1024xf16>
// CHECK:   }

}

// -----

// CHECK-LABEL: func.func @ReduceMeanSplitOverHWKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<680x1024x2x2xf16>) -> tensor<1x1x2x2xf16> {
func.func @ReduceMeanSplitOverHWKeepDimsTrue(%arg0: tensor<680x1024x2x2xf16>) -> tensor<1x1x2x2xf16> {
  %0 = VPU.ReduceMean(%arg0) {axes_value = [0, 1], keep_dims} : tensor<680x1024x2x2xf16> -> tensor<1x1x2x2xf16>
  return %0 : tensor<1x1x2x2xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [680, 1024, 1, 1] : tensor<680x1024x2x2xf16> to tensor<680x1024x1x1xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceMean(%[[VAL_1]]) {axes_value = [0, 1], keep_dims} : tensor<680x1024x1x1xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 1] [680, 1024, 1, 1] : tensor<680x1024x2x2xf16> to tensor<680x1024x1x1xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceMean(%[[VAL_3]]) {axes_value = [0, 1], keep_dims} : tensor<680x1024x1x1xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 1, 0] [680, 1024, 1, 1] : tensor<680x1024x2x2xf16> to tensor<680x1024x1x1xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceMean(%[[VAL_5]]) {axes_value = [0, 1], keep_dims} : tensor<680x1024x1x1xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 1, 1] [680, 1024, 1, 1] : tensor<680x1024x2x2xf16> to tensor<680x1024x1x1xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceMean(%[[VAL_7]]) {axes_value = [0, 1], keep_dims} : tensor<680x1024x1x1xf16> -> tensor<1x1x1x1xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x2xf16>
// CHECK:   return %[[VAL_9]] : tensor<1x1x2x2xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceMinSplitOverNCKeepDimsFalse(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x2xf16> {
func.func @ReduceMinSplitOverNCKeepDimsFalse(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x2xf16> {
  %0 = VPU.ReduceMin(%arg0) {axes_value = [2, 3]} : tensor<2x2x680x1024xf16> -> tensor<2x2xf16>
  return %0 : tensor<2x2xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceMin(%[[VAL_1]]) {axes_value = [2, 3]} : tensor<1x1x680x1024xf16> -> tensor<1x1xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 1, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceMin(%[[VAL_3]]) {axes_value = [2, 3]} : tensor<1x1x680x1024xf16> -> tensor<1x1xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [1, 0, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceMin(%[[VAL_5]]) {axes_value = [2, 3]} : tensor<1x1x680x1024xf16> -> tensor<1x1xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [1, 1, 0, 0] [1, 1, 680, 1024] : tensor<2x2x680x1024xf16> to tensor<1x1x680x1024xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceMin(%[[VAL_7]]) {axes_value = [2, 3]} : tensor<1x1x680x1024xf16> -> tensor<1x1xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0], [0, 1], [1, 0], [1, 1]]} : tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<1x1xf16> -> tensor<2x2xf16>
// CHECK:   return %[[VAL_9]] : tensor<2x2xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceProdSplitOverHKeepDimsFalse(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x680xf16> {
func.func @ReduceProdSplitOverHKeepDimsFalse(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x680xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3]} : tensor<2x2x680x1024xf16> -> tensor<2x680xf16>
  return %0 : tensor<2x680xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 227, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x227x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceProd(%[[VAL_1]]) {axes_value = [1, 3]} : tensor<2x2x227x1024xf16> -> tensor<2x227xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 227, 0] [2, 2, 227, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x227x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceProd(%[[VAL_3]]) {axes_value = [1, 3]} : tensor<2x2x227x1024xf16> -> tensor<2x227xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 454, 0] [2, 2, 226, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x226x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceProd(%[[VAL_5]]) {axes_value = [1, 3]} : tensor<2x2x226x1024xf16> -> tensor<2x226xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]]) {static_offsets = {{\[\[}}0, 0], [0, 227], [0, 454]]} : tensor<2x227xf16>, tensor<2x227xf16>, tensor<2x226xf16> -> tensor<2x680xf16>
// CHECK:   return %[[VAL_7]] : tensor<2x680xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceSumSplitOverWKeepDimsFalse(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x1024xf16> {
func.func @ReduceSumSplitOverWKeepDimsFalse(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x1024xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 2]} : tensor<2x2x680x1024xf16> -> tensor<2x1024xf16>
  return %0 : tensor<2x1024xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 680, 342] : tensor<2x2x680x1024xf16> to tensor<2x2x680x342xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [1, 2]} : tensor<2x2x680x342xf16> -> tensor<2x342xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 342] [2, 2, 680, 341] : tensor<2x2x680x1024xf16> to tensor<2x2x680x341xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceSum(%[[VAL_3]]) {axes_value = [1, 2]} : tensor<2x2x680x341xf16> -> tensor<2x341xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 683] [2, 2, 680, 341] : tensor<2x2x680x1024xf16> to tensor<2x2x680x341xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 2]} : tensor<2x2x680x341xf16> -> tensor<2x341xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]]) {static_offsets = {{\[\[}}0, 0], [0, 342], [0, 683]]} : tensor<2x342xf16>, tensor<2x341xf16>, tensor<2x341xf16> -> tensor<2x1024xf16>
// CHECK:   return %[[VAL_7]] : tensor<2x1024xf16>
// CHECK:   }
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
func.func @MVN1NormalizeSplit(%arg0: tensor<1x1x1x520001xf16>, %arg1: tensor<1x1x1x2xf16, {order = #NHWC}>) -> tensor<1x1x1x520001xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>
    return %0 : tensor<1x1x1x520001xf16>

    //CHECK:        [[INPUT_TILE_1:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 260001] : tensor<1x1x1x520001xf16> to tensor<1x1x1x260001xf16>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1Normalize([[INPUT_TILE_1]], %arg1)
    // CHECK-SAME:     :  tensor<1x1x1x260001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x260001xf16>

    //CHECK:        [[INPUT_TILE_2:%.+]] = VPU.Slice %arg0 [0, 0, 0, 260001] [1, 1, 1, 260000] : tensor<1x1x1x520001xf16> to tensor<1x1x1x260000xf16>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1Normalize([[INPUT_TILE_2]], %arg1)
    // CHECK-SAME:     :  tensor<1x1x1x260000xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x260000xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]])
    // CHECK:       return [[CONCAT]] :  tensor<1x1x1x520001xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtCWithInternalReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x8x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtCWithInternalReshape(%arg0: tensor<1x256x8x512xf32, {order = #NHWC}>) -> tensor<1x256x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = false, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [1, 256, 1024, 1024],
        internal_reshape = [1, 64, 4194304, 1],
        output_type = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x256x8x512xf32, {order = #NHWC}> -> tensor<1x256x1x2xf16, {order = #NHWC}>

    return %0 : tensor<1x256x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 88, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x88x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_1]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 22, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 88, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x88x8x512xf32, {order = #NHWC}> -> tensor<1x88x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 88, 0, 0] [1, 88, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x88x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_2]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 22, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 88, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x88x8x512xf32, {order = #NHWC}> -> tensor<1x88x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 176, 0, 0] [1, 80, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x80x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_3:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_3]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 20, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 80, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x80x8x512xf32, {order = #NHWC}> -> tensor<1x80x1x2xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]], [[OUTPUT_TILE_3]])

    // CHECK:       return [[CONCAT]] :  tensor<1x256x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtCWithoutInternalReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x8x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtCWithoutInternalReshape(%arg0: tensor<1x256x8x512xf32, {order = #NHWC}>) -> tensor<1x256x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = false, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [1, 256, 1024, 1024],
        output_type = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x256x8x512xf32, {order = #NHWC}> -> tensor<1x256x1x2xf16, {order = #NHWC}>

    return %0 : tensor<1x256x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 86, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x86x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_1]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 86, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x86x8x512xf32, {order = #NHWC}> -> tensor<1x86x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 86, 0, 0] [1, 85, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x85x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_2]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 85, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x85x8x512xf32, {order = #NHWC}> -> tensor<1x85x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 171, 0, 0] [1, 85, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x85x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_3:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_3]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 85, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x85x8x512xf32, {order = #NHWC}> -> tensor<1x85x1x2xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]], [[OUTPUT_TILE_3]])

    // CHECK:       return [[CONCAT]] :  tensor<1x256x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtNWithoutInternalReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x1x8x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtNWithoutInternalReshape(%arg0: tensor<256x1x8x512xf32, {order = #NHWC}>) -> tensor<256x1x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = true, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [256, 1, 1024, 1024],
        output_type = f16
    } : tensor<256x1x8x512xf32, {order = #NHWC}> -> tensor<256x1x1x2xf16, {order = #NHWC}>

    return %0 : tensor<256x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [86, 1, 8, 512] : tensor<256x1x8x512xf32, {order = #NHWC}> to tensor<86x1x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_1]]) {
    // CHECK-SAME:          across_channels = true, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [86, 1, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<86x1x8x512xf32, {order = #NHWC}> -> tensor<86x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [86, 0, 0, 0] [85, 1, 8, 512] : tensor<256x1x8x512xf32, {order = #NHWC}> to tensor<85x1x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_2]]) {
    // CHECK-SAME:          across_channels = true, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [85, 1, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<85x1x8x512xf32, {order = #NHWC}> -> tensor<85x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [171, 0, 0, 0] [85, 1, 8, 512] : tensor<256x1x8x512xf32, {order = #NHWC}> to tensor<85x1x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_3:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_3]]) {
    // CHECK-SAME:          across_channels = true, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [85, 1, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<85x1x8x512xf32, {order = #NHWC}> -> tensor<85x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]], [[OUTPUT_TILE_3]])

    // CHECK:       return [[CONCAT]] :  tensor<256x1x1x2xf16, {order = #NHWC}>
}
