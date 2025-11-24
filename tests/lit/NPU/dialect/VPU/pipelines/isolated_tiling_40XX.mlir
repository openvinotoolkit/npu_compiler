//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --tiling="prefetching=false" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

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
        strides = [1, 1]
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
        strides = [1, 1]
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
func.func @InterpSplitOverH(
    %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
            -> tensor<1x64x192x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [192, 320],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
      return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>
}

// CHECK:  [[SLICE0:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 0] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[SLICE0]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE1:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 11] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP1:%.+]] = VPU.Interpolate([[SLICE1]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE2:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 23] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP2:%.+]] = VPU.Interpolate([[SLICE2]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE3:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 34] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP3:%.+]] = VPU.Interpolate([[SLICE3]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE4:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 46] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP4:%.+]] = VPU.Interpolate([[SLICE4]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE5:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 57] [1, 64, 48, 13] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x13xf16, {order = #NHWC}>
// CHECK:  [[INTERP5:%.+]] = VPU.Interpolate([[SLICE5]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[SLICE6:%.+]] = VPU.Slice %arg0
// CHECK-SAME:  [0, 0, 0, 68] [1, 64, 48, 12] : tensor<1x64x48x80xf16, {order = #NHWC}> to tensor<1x64x48x12xf16, {order = #NHWC}>
// CHECK:  [[INTERP6:%.+]] = VPU.Interpolate([[SLICE6]])
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK:  [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]], [[INTERP3]], [[INTERP4]], [[INTERP5]], [[INTERP6]])
// CHECK:  return [[CONCAT]] : tensor<1x64x192x320xf16, {order = #NHWC}>

// -----

// CHECK-LABEL: func.func @InterpCubicSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x4x360x640xf16>
func.func @InterpCubicSplitOverC(
        %input1: tensor<1x4x360x640xf16>)
            -> tensor<1x4x144x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <TF_HALF_PIXEL_FOR_NN>, cube_coeff = -7.500000e-01, mode = <CUBIC>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [144, 256], scales_attr = [4.000000e-01, 4.000000e-01], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} :
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

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
func.func @InterpSplitOverCNoCommonFactor(
    %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
            -> tensor<1x64x121x121xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
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

!DistributedTensor0 = !VPU.DistributedTensor<
    1x32x100x100xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistributedTensor1 = !VPU.DistributedTensor<
    1x128x100x100xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>
// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT0:%.+]]: !VPU.DistributedTensor<1x32x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @NoTilingClusterNCEConv(%arg0: !DistributedTensor0) -> !DistributedTensor1 {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEStub<>,
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
                } : !DistributedTensor0, tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> -> !DistributedTensor1

    return %0 : !DistributedTensor1

    // CHECK-DAG:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>
    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:           [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[WEIGHTS]], [[WEIGHT_TABLE]])
    // CHECK-SAME:              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:         return [[NCE_CONV]] : !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
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

  // CHECK:     [[CST:%.+]] = const.Declare tensor<1x96768xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 290304], [1, 96768]>, #const.CastElemType<si32>]
  // CHECK:     [[CST0:%.+]] = const.Declare tensor<1x96768xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 193536], [1, 96768]>, #const.CastElemType<si32>]
  // CHECK:     [[CST1:%.+]] = const.Declare tensor<1x96768xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 96768], [1, 96768]>, #const.CastElemType<si32>]
  // CHECK:     [[CST2:%.+]] = const.Declare tensor<1x96768xsi32>
  // CHECK-SAME:tensor<1x387072xsi64>, [#const.SubView<[0, 0], [1, 96768]>, #const.CastElemType<si32>]

  // Tile 0
  // CHECK:     [[Tile0:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather0:%.+]] = VPU.Gather([[Tile0]], [[CST2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 1
  // CHECK:     [[Tile1:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather1:%.+]] = VPU.Gather([[Tile1]], [[CST2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 2
  // CHECK:     [[Tile2:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather2:%.+]] = VPU.Gather([[Tile2]], [[CST2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 3
  // CHECK:     [[Tile3:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather3:%.+]] = VPU.Gather([[Tile3]], [[CST1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 4
  // CHECK:     [[Tile4:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather4:%.+]] = VPU.Gather([[Tile4]], [[CST1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 5
  // CHECK:     [[Tile5:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather5:%.+]] = VPU.Gather([[Tile5]], [[CST1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 6
  // CHECK:     [[Tile6:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather6:%.+]] = VPU.Gather([[Tile6]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 7
  // CHECK:     [[Tile7:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather7:%.+]] = VPU.Gather([[Tile7]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 8
  // CHECK:     [[Tile8:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather8:%.+]] = VPU.Gather([[Tile8]], [[CST0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 9
  // CHECK:     [[Tile9:%.+]] = VPU.Slice [[INPUT]] [0, 0] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather9:%.+]] = VPU.Gather([[Tile9]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 10
  // CHECK:     [[Tile10:%.+]] = VPU.Slice [[INPUT]] [0, 1] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather10:%.+]] = VPU.Gather([[Tile10]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // Tile 11
  // CHECK:     [[Tile11:%.+]] = VPU.Slice [[INPUT]] [0, 2] [387072, 1] : tensor<387072x3xf16> to tensor<387072x1xf16>
  // CHECK:     [[Gather11:%.+]] = VPU.Gather([[Tile11]], [[CST]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x1xf16>, tensor<1x96768xsi32> -> tensor<1x96768x1xf16>

  // CHECK:    [[Concat:%.+]] = VPU.Concat([[Gather0]], [[Gather1]], [[Gather2]], [[Gather3]], [[Gather4]], [[Gather5]], [[Gather6]], [[Gather7]], [[Gather8]], [[Gather9]], [[Gather10]], [[Gather11]])
  // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 96768, 0], [0, 96768, 1], [0, 96768, 2], [0, 193536, 0], [0, 193536, 1], [0, 193536, 2], [0, 290304, 0], [0, 290304, 1], [0, 290304, 2]]}
  // CHECK-SAME{LITERAL}: tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16>, tensor<1x96768x1xf16> -> tensor<1x387072x3xf16>

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
    // CHECK:     [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2016, 320, 1] : tensor<1x4004x320x1xf16> to tensor<1x2016x320x1xf16>
    // CHECK:     [[GATHER0:%.+]] = VPU.Gather([[SLICE0]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x2016x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x2016x1x1xf16>

    // CHECK:     [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 2016, 0, 0] [1, 1988, 320, 1] : tensor<1x4004x320x1xf16> to tensor<1x1988x320x1xf16>
    // CHECK:     [[GATHER1:%.+]] = VPU.Gather([[SLICE1]], [[INDICES]]) {axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1988x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1988x1x1xf16>
    // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[GATHER0]], [[GATHER1]]) {
    // CHECK-SAME{LITERAL}:             static_offsets = [[0, 0, 0, 0], [0, 2016, 0, 0]]} : tensor<1x2016x1x1xf16>, tensor<1x1988x1x1xf16> -> tensor<1x4004x1x1xf16>

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

// CHECK-LABEL: func.func @Yuv2RGBSplit
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x993x736x1xf16>
func.func @Yuv2RGBSplit(%arg0: tensor<1x993x736x1xf16>) -> tensor<1x662x736x3xf16> {
  %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x662x736x1xf16>
  %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x331x736x1xf16>
  %2 = VPU.Reshape(%1) {shape_value = [1, 331, 368, 2]} : tensor<1x331x736x1xf16> -> tensor<1x331x368x2xf16>
  %3 = VPU.YuvToRgb(%0, %2) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x662x736x1xf16>, tensor<1x331x368x2xf16> -> tensor<1x662x736x3xf16>
  return %3 : tensor<1x662x736x3xf16>

    // CHECK:    [[UV_INPUT:%.+]] = VPU.Slice [[INPUT]] [0, 662, 0, 0] [1, 331, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x331x736x1xf16>
    // CHECK:    [[UV_CAST:%.+]] = VPU.ShapeCast {shape = [1, 331, 368, 2]} inputs([[UV_INPUT]] : tensor<1x331x736x1xf16>) -> tensor<1x331x368x2xf16>

    // Tile 0
    // CHECK:    [[Y0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 166, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x166x736x1xf16>
    // CHECK:    [[UV0:%.+]] = VPU.Slice [[UV_CAST]] [0, 0, 0, 0] [1, 83, 368, 2] : tensor<1x331x368x2xf16> to tensor<1x83x368x2xf16>
    // CHECK:    [[RGB0:%.+]] = VPU.YuvToRgb([[Y0]], [[UV0]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x166x736x1xf16>, tensor<1x83x368x2xf16> -> tensor<1x166x736x3xf16>

    // Tile 1
    // CHECK:    [[Y1:%.+]] = VPU.Slice [[INPUT]] [0, 166, 0, 0] [1, 166, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x166x736x1xf16>
    // CHECK:    [[UV1:%.+]] = VPU.Slice [[UV_CAST]] [0, 83, 0, 0] [1, 83, 368, 2] : tensor<1x331x368x2xf16> to tensor<1x83x368x2xf16>
    // CHECK:    [[RGB1:%.+]] = VPU.YuvToRgb([[Y1]], [[UV1]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x166x736x1xf16>, tensor<1x83x368x2xf16> -> tensor<1x166x736x3xf16>

    // Tile 2
    // CHECK:    [[Y2:%.+]] = VPU.Slice [[INPUT]] [0, 332, 0, 0] [1, 166, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x166x736x1xf16>
    // CHECK:    [[UV2:%.+]] = VPU.Slice [[UV_CAST]] [0, 166, 0, 0] [1, 83, 368, 2] : tensor<1x331x368x2xf16> to tensor<1x83x368x2xf16>
    // CHECK:    [[RGB2:%.+]] = VPU.YuvToRgb([[Y2]], [[UV2]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x166x736x1xf16>, tensor<1x83x368x2xf16> -> tensor<1x166x736x3xf16>

    // Tile 3
    // CHECK:    [[Y3:%.+]] = VPU.Slice [[INPUT]] [0, 498, 0, 0] [1, 164, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x164x736x1xf16>
    // CHECK:    [[UV3:%.+]] = VPU.Slice [[UV_CAST]] [0, 249, 0, 0] [1, 82, 368, 2] : tensor<1x331x368x2xf16> to tensor<1x82x368x2xf16>
    // CHECK:    [[RGB3:%.+]] = VPU.YuvToRgb([[Y3]], [[UV3]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x164x736x1xf16>, tensor<1x82x368x2xf16> -> tensor<1x164x736x3xf16>

    // CHECK:    [[CONCAT:%.+]] = VPU.Concat([[RGB0]], [[RGB1]], [[RGB2]], [[RGB3]])
    // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 166, 0, 0], [0, 332, 0, 0], [0, 498, 0, 0]]} : tensor<1x166x736x3xf16>, tensor<1x166x736x3xf16>, tensor<1x166x736x3xf16>, tensor<1x164x736x3xf16> -> tensor<1x662x736x3xf16>
    // CHECK:    return [[CONCAT]] : tensor<1x662x736x3xf16>
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
func.func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<300000x2xf16> {
    %cst = const.Declare tensor<300000x1xsi32> = dense<1> : tensor<300000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<300000x1xsi32> -> tensor<300000x2xf16>
    return %0 : tensor<300000x2xf16>

    // CHECK-DAG: [[Indices_1:%.+]] = const.Declare tensor<150000x1xsi32> = dense<1> : tensor<300000x1xsi32>, [#const.SubView<[150000, 0], [150000, 1]>]
    // CHECK-DAG: [[Indices_0:%.+]] = const.Declare tensor<150000x1xsi32> = dense<1> : tensor<300000x1xsi32>, [#const.SubView<[0, 0], [150000, 1]>]

    // CHECK: [[GatherND0:%.+]] = VPU.GatherND(%arg0, [[Indices_0]]) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<150000x1xsi32> -> tensor<150000x2xf16>
    // CHECK: [[GatherND1:%.+]] = VPU.GatherND(%arg0, [[Indices_1]]) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<150000x1xsi32> -> tensor<150000x2xf16>

    // CHECK: [[OUTPUT:%.+]] = VPU.Concat([[GatherND0]], [[GatherND1]])
    // CHECK-SAME: [0, 0], [150000, 0]
    // CHECK-SAME: -> tensor<300000x2xf16>

    // CHECK: return [[OUTPUT]] : tensor<300000x2xf16>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceBlocksFirstSplit
func.func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
  return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %2 = VPU.Slice %arg0 [0, 160, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %4 = VPU.Slice %arg0 [0, 320, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %5 = VPU.Convert(%4) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %6 = VPU.Concat(%1, %3, %5) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 160, 0, 0], [0, 320, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16, {order = #NHWC}>

  // CHECK:     %7 = VPU.Slice %6 [0, 0, 0, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %9 = VPU.Slice %6 [0, 0, 5, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %10 = VPU.DepthToSpace(%9) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %11 = VPU.Concat(%8, %10)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16, {order = #NHWC}>

  // CHECK:     %12 = VPU.Slice %11 [0, 0, 0, 0] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %14 = VPU.Slice %11 [0, 0, 0, 160] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %15 = VPU.Convert(%14) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %16 = VPU.Slice %11 [0, 0, 0, 320] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %17 = VPU.Convert(%16) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %18 = VPU.Concat(%13, %15, %17)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 160], [0, 0, 0, 320]
  // CHECK-SAME:-> tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     return %18 : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplit
func.func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
  %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
  %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
  return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %2 = VPU.Slice %arg0 [0, 160, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %4 = VPU.Slice %arg0 [0, 320, 0, 0] [1, 160, 10, 120] : tensor<1x480x10x120xf32, {order = #NHWC}> to tensor<1x160x10x120xf32, {order = #NHWC}>
  // CHECK:     %5 = VPU.Convert(%4) {dstElemType = f16} : tensor<1x160x10x120xf32, {order = #NHWC}> -> tensor<1x160x10x120xf16, {order = #NHWC}>

  // CHECK:     %6 = VPU.Concat(%1, %3, %5) {
  // CHECK-SAME:[0, 0, 0, 0], [0, 160, 0, 0], [0, 320, 0, 0]
  // CHECK-SAME:-> tensor<1x480x10x120xf16, {order = #NHWC}>

  // CHECK:     %7 = VPU.Slice %6 [0, 0, 0, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %8 = VPU.DepthToSpace(%7) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %9 = VPU.Slice %6 [0, 0, 5, 0] [1, 480, 5, 120] : tensor<1x480x10x120xf16, {order = #NHWC}> to tensor<1x480x5x120xf16, {order = #NHWC}>
  // CHECK:     %10 = VPU.DepthToSpace(%9) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x5x120xf16, {order = #NHWC}> -> tensor<1x30x20x480xf16, {order = #NHWC}>

  // CHECK:     %11 = VPU.Concat(%8, %10)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 20, 0]
  // CHECK-SAME:-> tensor<1x30x40x480xf16, {order = #NHWC}>

  // CHECK:     %12 = VPU.Slice %11 [0, 0, 0, 0] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %14 = VPU.Slice %11 [0, 0, 0, 160] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %15 = VPU.Convert(%14) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %16 = VPU.Slice %11 [0, 0, 0, 320] [1, 30, 40, 160] : tensor<1x30x40x480xf16, {order = #NHWC}> to tensor<1x30x40x160xf16, {order = #NHWC}>
  // CHECK:     %17 = VPU.Convert(%16) {dstElemType = f32} : tensor<1x30x40x160xf16, {order = #NHWC}> -> tensor<1x30x40x160xf32, {order = #NHWC}>

  // CHECK:     %18 = VPU.Concat(%13, %15, %17)
  // CHECK-SAME:[0, 0, 0, 0], [0, 0, 0, 160], [0, 0, 0, 320]
  // CHECK-SAME:-> tensor<1x30x40x480xf32, {order = #NHWC}>

  // CHECK:     return %18 : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @SpaceToDepthBlockFirstSplit
func.func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf16>) -> tensor<1x768x40x20xf16> {
      %0 = VPU.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
      return %0 : tensor<1x768x40x20xf16>

    // CHECK:     %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>
    // CHECK:     %2 = VPU.Slice %arg0 [0, 0, 0, 40] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:     %3 = VPU.SpaceToDepthOp(%2) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>
    // CHECK{LITERAL}:     %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 10]]} : tensor<1x768x40x10xf16>, tensor<1x768x40x10xf16> -> tensor<1x768x40x20xf16>

}


// -----

// CHECK-LABEL: func.func @SpaceToDepthDepthFirstSplit
func.func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 48, 54, 80] : tensor<1x48x160x80xf32> to tensor<1x48x54x80xf32>
    // CHECK:    %1 = VPU.Convert(%0) {dstElemType = f16} : tensor<1x48x54x80xf32> -> tensor<1x48x54x80xf16>
    // CHECK:    %2 = VPU.Slice %arg0 [0, 0, 54, 0] [1, 48, 53, 80] : tensor<1x48x160x80xf32> to tensor<1x48x53x80xf32>
    // CHECK:    %3 = VPU.Convert(%2) {dstElemType = f16} : tensor<1x48x53x80xf32> -> tensor<1x48x53x80xf16>
    // CHECK:    %4 = VPU.Slice %arg0 [0, 0, 107, 0] [1, 48, 53, 80] : tensor<1x48x160x80xf32> to tensor<1x48x53x80xf32>
    // CHECK:    %5 = VPU.Convert(%4) {dstElemType = f16} : tensor<1x48x53x80xf32> -> tensor<1x48x53x80xf16>
    // CHECK{LITERAL}:    %6 = VPU.Concat(%1, %3, %5) {static_offsets = [[0, 0, 0, 0], [0, 0, 54, 0], [0, 0, 107, 0]]} : tensor<1x48x54x80xf16>, tensor<1x48x53x80xf16>, tensor<1x48x53x80xf16> -> tensor<1x48x160x80xf16>
    // CHECK:    %7 = VPU.Slice %6 [0, 0, 0, 0] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:    %8 = VPU.SpaceToDepthOp(%7) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>
    // CHECK:    %9 = VPU.Slice %6 [0, 0, 0, 40] [1, 48, 160, 40] : tensor<1x48x160x80xf16> to tensor<1x48x160x40xf16>
    // CHECK:    %10 = VPU.SpaceToDepthOp(%9) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x40xf16> -> tensor<1x768x40x10xf16>
    // CHECK{LITERAL}:    %11 = VPU.Concat(%8, %10) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 10]]} : tensor<1x768x40x10xf16>, tensor<1x768x40x10xf16> -> tensor<1x768x40x20xf16>
    // CHECK:    %12 = VPU.Slice %11 [0, 0, 0, 0] [1, 256, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x256x40x20xf16>
    // CHECK:    %13 = VPU.Convert(%12) {dstElemType = f32} : tensor<1x256x40x20xf16> -> tensor<1x256x40x20xf32>
    // CHECK:    %14 = VPU.Slice %11 [0, 256, 0, 0] [1, 256, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x256x40x20xf16>
    // CHECK:    %15 = VPU.Convert(%14) {dstElemType = f32} : tensor<1x256x40x20xf16> -> tensor<1x256x40x20xf32>
    // CHECK:    %16 = VPU.Slice %11 [0, 512, 0, 0] [1, 256, 40, 20] : tensor<1x768x40x20xf16> to tensor<1x256x40x20xf16>
    // CHECK:    %17 = VPU.Convert(%16) {dstElemType = f32} : tensor<1x256x40x20xf16> -> tensor<1x256x40x20xf32>
    // CHECK{LITERAL}:    %18 = VPU.Concat(%13, %15, %17) {static_offsets = [[0, 0, 0, 0], [0, 256, 0, 0], [0, 512, 0, 0]]} : tensor<1x256x40x20xf32>, tensor<1x256x40x20xf32>, tensor<1x256x40x20xf32> -> tensor<1x768x40x20xf32>
    // CHECK:    return %18 : tensor<1x768x40x20xf32>
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
        } : tensor<1x32x64x48xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x48xf16, {order = #NHWC}>

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
    // CHECK-SAME:          strides = [1, 1]}
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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x64x64x!qElemType, {order = #NHWC}>, tensor<512x32x3x3x!qElemType2, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NCHW}> -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        [[CONST0:%.+]] = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    // CHECK:        [[CONST1:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>
    // CHECK:        [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64] : tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x32x33x64x!qElemType, {order = #NHWC}>
    // CHECK:        [[OUTPUT0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[CONST0]], [[CONST1]])
	// CHECK-SAME:        {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [512, 32, 3, 3], strides = [1, 1]}
	// CHECK-SAME:        -> tensor<1x512x32x64x!qElemType1, {order = #NHWC}>
    // CHECK:        [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64] : tensor<1x32x64x64x!qElemType, {order = #NHWC}> to tensor<1x32x33x64x!qElemType, {order = #NHWC}>
    // CHECK:        [[OUTPUT1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[CONST0]], [[CONST1]])
	// CHECK-SAME:        {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [512, 32, 3, 3], strides = [1, 1]}
	// CHECK-SAME:        -> tensor<1x512x32x64x!qElemType1, {order = #NHWC}>
    // CHECK:        [[CONCAT:%.+]] = VPU.Concat(%1, %3)
	// CHECK-LITERAL:        {static_offsets = [[0, 0, 0, 0], [0, 0, 32, 0]]}
	// CHECK-SAME:        tensor<1x512x32x64x!qElemType1, {order = #NHWC}>, tensor<1x512x32x64x!qElemType1, {order = #NHWC}> -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
    // CHECK:        return [[CONCAT]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
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
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x4xsi32, {order = #NCHW}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<3168x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [3168, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE_0:%.+]] = const.Declare tensor<3168x1x1x4xsi32> = dense<10> : tensor<6320x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [3168, 1, 1, 4]>]

    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<3152x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.SubView<[3168, 0, 0, 0], [3152, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE_1:%.+]] = const.Declare tensor<3152x1x1x4xsi32> = dense<10> : tensor<6320x1x1x4xsi32>, [#const.SubView<[3168, 0, 0, 0], [3152, 1, 1, 4]>]

    // CHECK:           [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]], [[WEIGHTS_TABLE_0]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [3168, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x3168x256x4xf16, {order = #NHWC}>

    // CHECK:           [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_1]], [[WEIGHTS_TABLE_1]]) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [3152, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x3152x256x4xf16, {order = #NHWC}>

    // CHECK:           [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]]) {
    // CHECK-SAME{LITERAL}: static_offsets = [[0, 0, 0, 0], [0, 3168, 0, 0]]}
    // CHECK-SAME:      : tensor<1x3168x256x4xf16, {order = #NHWC}>, tensor<1x3152x256x4xf16, {order = #NHWC}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:           return [[CONCAT]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
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
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x16xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x16xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x16xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x16xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x1024x24x16xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x16xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT0_TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 24, 16]
    // CHECK-SAME:      : tensor<1x1024x24x16xf16, {order = #NHWC}> to tensor<1x512x24x16xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 512, 24, 16]
    // CHECK-SAME:      : tensor<1x1024x24x16xf16, {order = #NHWC}> to tensor<1x512x24x16xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE0]], [[INPUT1_TILE0]])
    // CHECK-SAME:      -> tensor<1x512x24x16xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT0_TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 512, 0, 0] [1, 512, 24, 16]
    // CHECK-SAME:      : tensor<1x1024x24x16xf16, {order = #NHWC}> to tensor<1x512x24x16xf16, {order = #NHWC}>

    // CHECK:       [[INPUT1_TILE1:%.+]] = VPU.Slice [[INPUT2]] [0, 512, 0, 0] [1, 512, 24, 16]
    // CHECK-SAME:      : tensor<1x1024x24x16xf16, {order = #NHWC}> to tensor<1x512x24x16xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Eltwise([[INPUT0_TILE1]], [[INPUT1_TILE1]])
    // CHECK-SAME:      -> tensor<1x512x24x16xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:      -> tensor<1x1024x24x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x24x16xf16, {order = #NHWC}>
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

func.func @ConvertU8F32SplitOverW(%arg0: tensor<1x2x80x3000xui8, {order = #NHWC}>) -> tensor<1x2x80x3000xf32, {order = #NHWC}> {
  %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x3000xui8, {order = #NHWC}> -> tensor<1x2x80x3000xf32, {order = #NHWC}>
  return %0 : tensor<1x2x80x3000xf32, {order = #NHWC}>
}

// CHECK-LABEL: @ConvertU8F32SplitOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x3000xui8, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 2, 80, 1500]
// CHECK-SAME:      : tensor<1x2x80x3000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x80x1500xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convert([[INPUT_TILE0]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x80x1500xf32, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1500] [1, 2, 80, 1500]
// CHECK-SAME:      : tensor<1x2x80x3000xui8, {order = #NHWC}>
// CHECK-SAME:      to tensor<1x2x80x1500xui8, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convert([[INPUT_TILE1]]) {
// CHECK-SAME:      dstElemType = f32
// CHECK-SAME:      }> -> tensor<1x2x80x1500xf32, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 1500]
// CHECK-SAME:      -> tensor<1x2x80x3000xf32, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x2x80x3000xf32, {order = #NHWC}>

// -----

func.func @SigmoidSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @SigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sigmoid([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sigmoid([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @TanhSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Tanh(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @TanhSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tanh([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tanh([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @ExpSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Exp(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @ExpSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Exp([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Exp([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @SqrtSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @SqrtSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sqrt([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sqrt([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
// -----

func.func @EluSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @EluSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Elu([[INPUT_TILE0]]) {
// CHECK-SAME:    x = 1.000000e+00 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Elu([[INPUT_TILE1]]) {
// CHECK-SAME:    x = 1.000000e+00 : f64} : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @ClampSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 8, 80, 480]
// CHECK-SAME:  : tensor<1x8x80x960xf16> to tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x480xf16> -> tensor<1x8x80x480xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 480]
// CHECK-SAME:  : tensor<1x8x80x480xf16>, tensor<1x8x80x480xf16> -> tensor<1x8x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>

// -----

func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
  return %0 : tensor<1x8x80x960xf16>
}

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {

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

// -----

func.func @HSwishSplitOverW(%arg0: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16> {
  %0 = VPU.HSwish(%arg0) : tensor<1x16x80x960xf16> -> tensor<1x16x80x960xf16>
  return %0 : tensor<1x16x80x960xf16>
}

// CHECK-LABEL: @HSwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 80, 240]
// CHECK-SAME:  : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSwish([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 240] [1, 16, 80, 240]
// CHECK-SAME:  : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSwish([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 480] [1, 16, 80, 240]
// CHECK-SAME:  : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.HSwish([[INPUT_TILE2]])
// CHECK-SAME:  : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 720] [1, 16, 80, 240]
// CHECK-SAME:  : tensor<1x16x80x960xf16> to tensor<1x16x80x240xf16>
// CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.HSwish([[INPUT_TILE3]])
// CHECK-SAME:  : tensor<1x16x80x240xf16> -> tensor<1x16x80x240xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 240], [0, 0, 0, 480], [0, 0, 0, 720]
// CHECK-SAME:  : tensor<1x16x80x240xf16>, tensor<1x16x80x240xf16>, tensor<1x16x80x240xf16>,
// CHECK-SAME:   tensor<1x16x80x240xf16> -> tensor<1x16x80x960xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x16x80x960xf16>

// -----

func.func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
  %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
  return %0 : tensor<1x10x256x176xf16>
}

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Divide([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 176]
// CHECK-SAME:   : tensor<1x10x256x176xf16> to tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Divide([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x128x176xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x176xf16>, tensor<1x10x128x176xf16> -> tensor<1x10x256x176xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x30x40xf16>) -> tensor<1x30x40x546xf16> {
  %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x30x40xf16> -> tensor<1x30x40x546xf16>
  return %0 : tensor<1x30x40x546xf16>
}
// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x30x40xf16>) -> tensor<1x30x40x546xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 273, 30, 40]
// CHECK-SAME:  : tensor<1x546x30x40xf16> to tensor<1x273x30x40xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MemPermute([[INPUT_TILE0]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC
// CHECK-SAME:  } : tensor<1x273x30x40xf16> -> tensor<1x30x40x273xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 273, 0, 0] [1, 273, 30, 40]
// CHECK-SAME:  : tensor<1x546x30x40xf16> to tensor<1x273x30x40xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MemPermute([[INPUT_TILE1]]) {
// CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC
// CHECK-SAME:  } : tensor<1x273x30x40xf16> -> tensor<1x30x40x273xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 273]
// CHECK-SAME:  : tensor<1x30x40x273xf16>, tensor<1x30x40x273xf16> -> tensor<1x30x40x546xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x30x40x546xf16>

// -----

func.func @AvgPoolSwSplit2Part(%arg0: tensor<1x24x1800x16xf16>) -> tensor<1x24x1789x16xf16> {
  %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x24x1800x16xf16> -> tensor<1x24x1789x16xf16>
  return %0 : tensor<1x24x1789x16xf16>
}
// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x24x1800x16xf16>) -> tensor<1x24x1789x16xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 24, 906, 16]
// CHECK-SAME:  :  tensor<1x24x1800x16xf16> to tensor<1x24x906x16xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
// CHECK-SAME:  } : tensor<1x24x906x16xf16> -> tensor<1x24x895x16xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 895, 0] [1, 24, 905, 16]
// CHECK-SAME:  : tensor<1x24x1800x16xf16> to tensor<1x24x905x16xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]]) {
// CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]
// CHECK-SAME:  } : tensor<1x24x905x16xf16> -> tensor<1x24x894x16xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 895, 0]
// CHECK-SAME:  : tensor<1x24x895x16xf16>, tensor<1x24x894x16xf16> -> tensor<1x24x1789x16xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x24x1789x16xf16>

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
    } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>, tensor<160x1x1x4xsi32, {order = #NCHW}> -> tensor<1x160x80x60xf16, {order = #NHWC}>

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
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x80x80x!qElemType, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>, tensor<320x1x1x4xsi32, {order = #NCHW}> -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

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
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x12960xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x12960xf16, {order = #NHWC}>) -> tensor<1x16x1x12960xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x12960xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x12960xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 4320] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 8640] [1, 16, 7, 4320]
    // CHECK-SAME:      tensor<1x16x7x12960xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4320xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 4320], [0, 0, 0, 8640]
    // CHECK-SAME:      -> tensor<1x16x1x12960xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x12960xf16, {order = #NHWC}>
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

// CHECK-LABEL: @ReduceSumTwoAxesKeepDimsFalse
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x784x64x32xf16>

func.func @ReduceSumTwoAxesKeepDimsFalse(%arg0: tensor<1x784x64x32xf16>) -> tensor<1x64xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 3]} : tensor<1x784x64x32xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>

// CHECK:    [[SLICE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 784, 22, 32] : tensor<1x784x64x32xf16> to tensor<1x784x22x32xf16>
// CHECK:    [[REDUCE_SUM0:%.+]] = VPU.ReduceSum([[SLICE0]]) {axes_value = [1, 3]} : tensor<1x784x22x32xf16> -> tensor<1x22xf16>
// CHECK:    [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 22, 0] [1, 784, 21, 32] : tensor<1x784x64x32xf16> to tensor<1x784x21x32xf16>
// CHECK:    [[REDUCE_SUM1:%.+]] = VPU.ReduceSum([[SLICE1]]) {axes_value = [1, 3]} : tensor<1x784x21x32xf16> -> tensor<1x21xf16>
// CHECK:    [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 43, 0] [1, 784, 21, 32] : tensor<1x784x64x32xf16> to tensor<1x784x21x32xf16>
// CHECK:    [[REDUCE_SUM2:%.+]] = VPU.ReduceSum([[SLICE2]]) {axes_value = [1, 3]} : tensor<1x784x21x32xf16> -> tensor<1x21xf16>
// CHECK:    [[CONCAT:%.+]] = VPU.Concat([[REDUCE_SUM0]], [[REDUCE_SUM1]], [[REDUCE_SUM2]])
// CHECK-SAME{LITERAL}:   {static_offsets = [[0, 0], [0, 22], [0, 43]]} : tensor<1x22xf16>, tensor<1x21xf16>, tensor<1x21xf16> -> tensor<1x64xf16>
// CHECK:    return [[CONCAT]] : tensor<1x64xf16>
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

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceLogicalOr(%[[VAL_1]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x170x1024xf16> -> tensor<2x1x170x1xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 170, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceLogicalOr(%[[VAL_3]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x170x1024xf16> -> tensor<2x1x170x1xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 340, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceLogicalOr(%[[VAL_5]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x170x1024xf16> -> tensor<2x1x170x1xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 510, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceLogicalOr(%[[VAL_7]]) {axes_value = [1, 3], keep_dims} : tensor<2x2x170x1024xf16> -> tensor<2x1x170x1xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 170, 0], [0, 0, 340, 0], [0, 0, 510, 0]]} : tensor<2x1x170x1xf16>, tensor<2x1x170x1xf16>, tensor<2x1x170x1xf16>, tensor<2x1x170x1xf16> -> tensor<2x1x680x1xf16>
// CHECK:   return %[[VAL_9]] : tensor<2x1x680x1xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceMaxSplitOverWKeepDimsTrue(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x1x1x1024xf16> {
func.func @ReduceMaxSplitOverWKeepDimsTrue(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x1x1x1024xf16> {
  %0 = VPU.ReduceMax(%arg0) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x1024xf16> -> tensor<2x1x1x1024xf16>
  return %0 : tensor<2x1x1x1024xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceMax(%[[VAL_1]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x256xf16> -> tensor<2x1x1x256xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 256] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceMax(%[[VAL_3]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x256xf16> -> tensor<2x1x1x256xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 512] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceMax(%[[VAL_5]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x256xf16> -> tensor<2x1x1x256xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 768] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceMax(%[[VAL_7]]) {axes_value = [1, 2], keep_dims} : tensor<2x2x680x256xf16> -> tensor<2x1x1x256xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 256], [0, 0, 0, 512], [0, 0, 0, 768]]} : tensor<2x1x1x256xf16>, tensor<2x1x1x256xf16>, tensor<2x1x1x256xf16>, tensor<2x1x1x256xf16> -> tensor<2x1x1x1024xf16>
// CHECK:   return %[[VAL_9]] : tensor<2x1x1x1024xf16>
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
// CHECK-SAME:      %[[VAL_0:.+]]: tensor<2x2x680x1024xf16>) -> tensor<2x680xf16> {
func.func @ReduceProdSplitOverHKeepDimsFalse(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x680xf16> {
  %0 = VPU.ReduceProd(%arg0) {axes_value = [1, 3]} : tensor<2x2x680x1024xf16> -> tensor<2x680xf16>
  return %0 : tensor<2x680xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceProd(%[[VAL_1]]) {axes_value = [1, 3]} : tensor<2x2x170x1024xf16> -> tensor<2x170xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 170, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceProd(%[[VAL_3]]) {axes_value = [1, 3]} : tensor<2x2x170x1024xf16> -> tensor<2x170xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 340, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceProd(%[[VAL_5]]) {axes_value = [1, 3]} : tensor<2x2x170x1024xf16> -> tensor<2x170xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 510, 0] [2, 2, 170, 1024] : tensor<2x2x680x1024xf16> to tensor<2x2x170x1024xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceProd(%[[VAL_7]]) {axes_value = [1, 3]} : tensor<2x2x170x1024xf16> -> tensor<2x170xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0], [0, 170], [0, 340], [0, 510]]} : tensor<2x170xf16>, tensor<2x170xf16>, tensor<2x170xf16>, tensor<2x170xf16> -> tensor<2x680xf16>

// CHECK:   return %[[VAL_9]] : tensor<2x680xf16>
// CHECK:   }
}

// -----

// CHECK-LABEL: func.func @ReduceSumSplitOverWKeepDimsFalse(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x2x680x1024xf16>) -> tensor<2x1024xf16> {
func.func @ReduceSumSplitOverWKeepDimsFalse(%arg0: tensor<2x2x680x1024xf16>) -> tensor<2x1024xf16> {
  %0 = VPU.ReduceSum(%arg0) {axes_value = [1, 2]} : tensor<2x2x680x1024xf16> -> tensor<2x1024xf16>
  return %0 : tensor<2x1024xf16>

// CHECK:   %[[VAL_1:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 0] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_2:.*]] = VPU.ReduceSum(%[[VAL_1]]) {axes_value = [1, 2]} : tensor<2x2x680x256xf16> -> tensor<2x256xf16>
// CHECK:   %[[VAL_3:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 256] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_4:.*]] = VPU.ReduceSum(%[[VAL_3]]) {axes_value = [1, 2]} : tensor<2x2x680x256xf16> -> tensor<2x256xf16>
// CHECK:   %[[VAL_5:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 512] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_6:.*]] = VPU.ReduceSum(%[[VAL_5]]) {axes_value = [1, 2]} : tensor<2x2x680x256xf16> -> tensor<2x256xf16>
// CHECK:   %[[VAL_7:.*]] = VPU.Slice %[[VAL_0]] [0, 0, 0, 768] [2, 2, 680, 256] : tensor<2x2x680x1024xf16> to tensor<2x2x680x256xf16>
// CHECK:   %[[VAL_8:.*]] = VPU.ReduceSum(%[[VAL_7]]) {axes_value = [1, 2]} : tensor<2x2x680x256xf16> -> tensor<2x256xf16>
// CHECK:   %[[VAL_9:.*]] = VPU.Concat(%[[VAL_2]], %[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) {static_offsets = {{\[\[}}0, 0], [0, 256], [0, 512], [0, 768]]} : tensor<2x256xf16>, tensor<2x256xf16>, tensor<2x256xf16>, tensor<2x256xf16> -> tensor<2x1024xf16>
// CHECK:   return %[[VAL_9]] : tensor<2x1024xf16>
// CHECK:   }
}

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

module @executors {
config.Resources 2 of @NCE at 1.700000e+03 MHz

// CHECK: func.func @NCEAveragePoolWithLargeKernelSOH([[INPUT:%.+]]: tensor<1x64x256x256xf16, {order = #NHWC}>)
func.func @NCEAveragePoolWithLargeKernelSOH(%arg0: tensor<1x64x256x256xf16, {order = #NHWC}>) -> tensor<1x64x4x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
                kernel_size = [8, 8],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                strides = [8, 8],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
            } -> tensor<1x64x32x32xf16, {order = #NHWC}>
    %1 = VPU.NCE.AveragePool(%0) {
                kernel_size = [8, 8],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEStub<>,
                strides = [8, 8],
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
            } -> tensor<1x64x4x4xf16, {order = #NHWC}>

    return %1 : tensor<1x64x4x4xf16, {order = #NHWC}>

    // CHECK:        [[INPUT_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 64, 256] : tensor<1x64x256x256xf16, {order = #NHWC}> to tensor<1x64x64x256xf16, {order = #NHWC}>
    // CHECK:        [[AVG_POOL_0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE_0]]) {
    // CHECK-SAME:              kernel_size = [8, 8], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [8, 8]}
    // CHECK-SAME:           -> tensor<1x64x8x32xf16, {order = #NHWC}>

    // CHECK:        [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 64, 0] [1, 64, 64, 256] : tensor<1x64x256x256xf16, {order = #NHWC}> to tensor<1x64x64x256xf16, {order = #NHWC}>
    // CHECK:        [[AVG_POOL_1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE_1]]) {
    // CHECK-SAME:              kernel_size = [8, 8], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [8, 8]}
    // CHECK-SAME:           -> tensor<1x64x8x32xf16, {order = #NHWC}>

    // CHECK:        [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 64, 64, 256] : tensor<1x64x256x256xf16, {order = #NHWC}> to tensor<1x64x64x256xf16, {order = #NHWC}>
    // CHECK:        [[AVG_POOL_2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE_2]]) {
    // CHECK-SAME:              kernel_size = [8, 8], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [8, 8]}
    // CHECK-SAME:           -> tensor<1x64x8x32xf16, {order = #NHWC}>

    // CHECK:        [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 192, 0] [1, 64, 64, 256] : tensor<1x64x256x256xf16, {order = #NHWC}> to tensor<1x64x64x256xf16, {order = #NHWC}>
    // CHECK:        [[AVG_POOL_3:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE_3]]) {
    // CHECK-SAME:              kernel_size = [8, 8], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [8, 8]}
    // CHECK-SAME:           -> tensor<1x64x8x32xf16, {order = #NHWC}>

    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[AVG_POOL_0]], [[AVG_POOL_1]], [[AVG_POOL_2]], [[AVG_POOL_3]]) {
    // CHECK-SAME{LITERAL}:     static_offsets = [[0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 16, 0], [0, 0, 24, 0]]}
    // CHECK-SAME:              tensor<1x64x8x32xf16, {order = #NHWC}>, tensor<1x64x8x32xf16, {order = #NHWC}>, tensor<1x64x8x32xf16, {order = #NHWC}>, tensor<1x64x8x32xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x64x32x32xf16, {order = #NHWC}>

    // CHECK:        [[AVG_POOL:%.+]] = VPU.NCE.AveragePool([[CONCAT]]) {
    // CHECK-SAME:              kernel_size = [8, 8], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [8, 8]}
    // CHECK-SAME:           -> tensor<1x64x4x4xf16, {order = #NHWC}>

    // CHECK:        return [[AVG_POOL]] : tensor<1x64x4x4xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

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

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x64x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_1]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 16, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 64, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x64x8x512xf32, {order = #NHWC}> -> tensor<1x64x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [0, 64, 0, 0] [1, 64, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x64x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_2]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 16, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 64, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x64x8x512xf32, {order = #NHWC}> -> tensor<1x64x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_3:%.+]] = VPU.Slice [[INPUT]] [0, 128, 0, 0] [1, 64, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x64x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_3:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_3]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 16, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 64, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x64x8x512xf32, {order = #NHWC}> -> tensor<1x64x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_4:%.+]] = VPU.Slice [[INPUT]] [0, 192, 0, 0] [1, 64, 8, 512] : tensor<1x256x8x512xf32, {order = #NHWC}> to tensor<1x64x8x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_4:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_4]]) {
    // CHECK-SAME:          across_channels = false, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          internal_reshape = [1, 16, 4194304, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [1, 64, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<1x64x8x512xf32, {order = #NHWC}> -> tensor<1x64x1x2xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]], [[OUTPUT_TILE_3]], [[OUTPUT_TILE_4]])

    // CHECK:       return [[CONCAT]] :  tensor<1x256x1x2xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

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

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
config.Resources 4 of @NCE at 1.700000e+03 MHz

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtNWithoutInternalReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x1x4x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtNWithoutInternalReshape(%arg0: tensor<256x1x4x512xf32, {order = #NHWC}>) -> tensor<256x1x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = true, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [256, 1, 1024, 1024],
        output_type = f16
    } : tensor<256x1x4x512xf32, {order = #NHWC}> -> tensor<256x1x1x2xf16, {order = #NHWC}>

    return %0 : tensor<256x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [128, 1, 4, 512] : tensor<256x1x4x512xf32, {order = #NHWC}> to tensor<128x1x4x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_1:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_1]]) {
    // CHECK-SAME:          across_channels = true, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [128, 1, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<128x1x4x512xf32, {order = #NHWC}> -> tensor<128x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE_2:%.+]] = VPU.Slice [[INPUT]] [128, 0, 0, 0] [128, 1, 4, 512] : tensor<256x1x4x512xf32, {order = #NHWC}> to tensor<128x1x4x512xf32, {order = #NHWC}>
    // CHECK:       [[OUTPUT_TILE_2:%.+]] = VPU.MVN1MeanVar([[INPUT_TILE_2]]) {
    // CHECK-SAME:          across_channels = true, eps = 9.9999999747524271E-7 : f64,
    // CHECK-SAME:          normalize_variance = true, orig_shape = [128, 1, 1024, 1024], output_type = f16
    // CHECK-SAME:      } : tensor<128x1x4x512xf32, {order = #NHWC}> -> tensor<128x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[OUTPUT_TILE_1]], [[OUTPUT_TILE_2]])

    // CHECK:       return [[CONCAT]] :  tensor<256x1x1x2xf16, {order = #NHWC}>
}

}

// -----

// CHECK-LABEL: func.func @GatherNDSplitWithoutOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4x32x56x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x4x14580x2xsi32>
func.func @GatherNDSplitWithoutOriginalShape(%arg0: tensor<1x4x32x56x16xf16>, %arg1: tensor<1x4x14580x2xsi32>) -> tensor<1x4x14580x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64
                    } : tensor<1x4x32x56x16xf16>, tensor<1x4x14580x2xsi32> -> tensor<1x4x14580x16xf16>

    return %0 : tensor<1x4x14580x16xf16>

    // CHECK:       [[SLICE_IN_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0, 0] [1, 2, 32, 56, 16] : tensor<1x4x32x56x16xf16> to tensor<1x2x32x56x16xf16>
    // CHECK:       [[SLICE_INDICES_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 2, 14580, 2] : tensor<1x4x14580x2xsi32> to tensor<1x2x14580x2xsi32>
    // CHECK:       [[GATHER_0:%.+]] = VPU.GatherND([[SLICE_IN_0]], [[SLICE_INDICES_0]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:             : tensor<1x2x32x56x16xf16>, tensor<1x2x14580x2xsi32> -> tensor<1x2x14580x16xf16>

    // CHECK:       [[SLICE_IN_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 2, 0, 0, 0] [1, 2, 32, 56, 16] : tensor<1x4x32x56x16xf16> to tensor<1x2x32x56x16xf16>
    // CHECK:       [[SLICE_INDICES_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 2, 0, 0] [1, 2, 14580, 2] : tensor<1x4x14580x2xsi32> to tensor<1x2x14580x2xsi32>
    // CHECK:       [[GATHER_1:%.+]] = VPU.GatherND([[SLICE_IN_1]], [[SLICE_INDICES_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:             : tensor<1x2x32x56x16xf16>, tensor<1x2x14580x2xsi32> -> tensor<1x2x14580x16xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER_0]], [[GATHER_1]]) {
    // CHECK-SAME{LITERAL}:               static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    // CHECK-SAME:             : tensor<1x2x14580x16xf16>, tensor<1x2x14580x16xf16> -> tensor<1x4x14580x16xf16>

    // CHECK: return [[CONCAT]] : tensor<1x4x14580x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherND4DSplitWithOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4x1792x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x4x14580x2xsi32>
func.func @GatherND4DSplitWithOriginalShape(%arg0: tensor<1x4x1792x16xf16>, %arg1: tensor<1x4x14580x2xsi32>) -> tensor<1x4x14580x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64, original_shape = [1, 4, 32, 56, 16]
                    } : tensor<1x4x1792x16xf16>, tensor<1x4x14580x2xsi32> -> tensor<1x4x14580x16xf16>

    return %0 : tensor<1x4x14580x16xf16>

    // CHECK:       [[SLICE_IN_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 2, 1792, 16] : tensor<1x4x1792x16xf16> to tensor<1x2x1792x16xf16>
    // CHECK:       [[SLICE_INDICES_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 2, 14580, 2] : tensor<1x4x14580x2xsi32> to tensor<1x2x14580x2xsi32>
    // CHECK:       [[GATHER_0:%.+]] = VPU.GatherND([[SLICE_IN_0]], [[SLICE_INDICES_0]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:               original_shape = [1, 2, 32, 56, 16]
    // CHECK-SAME:             : tensor<1x2x1792x16xf16>, tensor<1x2x14580x2xsi32> -> tensor<1x2x14580x16xf16>

    // CHECK:       [[SLICE_IN_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 2, 0, 0] [1, 2, 1792, 16] : tensor<1x4x1792x16xf16> to tensor<1x2x1792x16xf16>
    // CHECK:       [[SLICE_INDICES_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 2, 0, 0] [1, 2, 14580, 2] : tensor<1x4x14580x2xsi32> to tensor<1x2x14580x2xsi32>
    // CHECK:       [[GATHER_1:%.+]] = VPU.GatherND([[SLICE_IN_1]], [[SLICE_INDICES_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:               original_shape = [1, 2, 32, 56, 16]
    // CHECK-SAME:             : tensor<1x2x1792x16xf16>, tensor<1x2x14580x2xsi32> -> tensor<1x2x14580x16xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER_0]], [[GATHER_1]]) {
    // CHECK-SAME{LITERAL}:               static_offsets = [[0, 0, 0, 0], [0, 2, 0, 0]]
    // CHECK-SAME:             : tensor<1x2x14580x16xf16>, tensor<1x2x14580x16xf16> -> tensor<1x4x14580x16xf16>

    // CHECK: return [[CONCAT]] : tensor<1x4x14580x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitAtIndicesWithoutOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x9x16x8xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x104580x2xsi32>
func.func @GatherNDSplitAtIndicesWithoutOriginalShape(%arg0: tensor<1x9x16x8xf16>, %arg1: tensor<1x104580x2xsi32>) -> tensor<1x104580x8xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 1 : i64
                    } : tensor<1x9x16x8xf16>, tensor<1x104580x2xsi32> -> tensor<1x104580x8xf16>

    return %0 : tensor<1x104580x8xf16>

    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0] [1, 52290, 2] : tensor<1x104580x2xsi32> to tensor<1x52290x2xsi32>
    // CHECK:       [[GATHER_0:%.+]] = VPU.GatherND([[INPUT_0]], [[SLICE_0]]) {
    // CHECK-SAME:               batch_dims = 1 : i64
    // CHECK-SAME:             : tensor<1x9x16x8xf16>, tensor<1x52290x2xsi32> -> tensor<1x52290x8xf16>

    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 52290, 0] [1, 52290, 2] : tensor<1x104580x2xsi32> to tensor<1x52290x2xsi32>
    // CHECK:       [[GATHER_1:%.+]] = VPU.GatherND([[INPUT_0]], [[SLICE_1]]) {
    // CHECK-SAME:               batch_dims = 1 : i64
    // CHECK-SAME:             : tensor<1x9x16x8xf16>, tensor<1x52290x2xsi32> -> tensor<1x52290x8xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER_0]], [[GATHER_1]]) {
    // CHECK-SAME{LITERAL}:               static_offsets = [[0, 0, 0], [0, 52290, 0]]
    // CHECK-SAME:             : tensor<1x52290x8xf16>, tensor<1x52290x8xf16> -> tensor<1x104580x8xf16>

    // CHECK: return [[CONCAT]] : tensor<1x104580x8xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitAtIndicesWithOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x144x8xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x104580x2xsi32>
func.func @GatherNDSplitAtIndicesWithOriginalShape(%arg0: tensor<1x1x144x8xf16>, %arg1: tensor<1x1x104580x2xsi32>) -> tensor<1x1x104580x8xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64, original_shape = [1, 1, 9, 16, 8]
                    } : tensor<1x1x144x8xf16>, tensor<1x1x104580x2xsi32> -> tensor<1x1x104580x8xf16>

    return %0 : tensor<1x1x104580x8xf16>

    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 1, 52290, 2] : tensor<1x1x104580x2xsi32> to tensor<1x1x52290x2xsi32>
    // CHECK:       [[GATHER_0:%.+]] = VPU.GatherND([[INPUT_0]], [[SLICE_0]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:               original_shape = [1, 1, 9, 16, 8]
    // CHECK-SAME:             : tensor<1x1x144x8xf16>, tensor<1x1x52290x2xsi32> -> tensor<1x1x52290x8xf16>

    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 52290, 0] [1, 1, 52290, 2] : tensor<1x1x104580x2xsi32> to tensor<1x1x52290x2xsi32>
    // CHECK:       [[GATHER_1:%.+]] = VPU.GatherND([[INPUT_0]], [[SLICE_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64
    // CHECK-SAME:               original_shape = [1, 1, 9, 16, 8]
    // CHECK-SAME:             : tensor<1x1x144x8xf16>, tensor<1x1x52290x2xsi32> -> tensor<1x1x52290x8xf16>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GATHER_0]], [[GATHER_1]]) {
    // CHECK-SAME{LITERAL}:               static_offsets = [[0, 0, 0, 0], [0, 0, 52290, 0]]
    // CHECK-SAME:             : tensor<1x1x52290x8xf16>, tensor<1x1x52290x8xf16> -> tensor<1x1x104580x8xf16>

    // CHECK: return [[CONCAT]] : tensor<1x1x104580x8xf16>
}
