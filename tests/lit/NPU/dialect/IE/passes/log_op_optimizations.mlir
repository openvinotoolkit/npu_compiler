//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --log-op-optimizations %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @InterpolateNearest(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x12x12xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <NEAREST>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 12]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x12x12xf16>

    return %0 : tensor<1x3x12x12xf16>

    // CHECK: Interpolate at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @InterpolateNearestFloatScales(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x15x15xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <NEAREST>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.500000e+00, 2.500000e+00], sizes_attr = [15, 15]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x15x15xf16>

    return %0 : tensor<1x3x15x15xf16>

    // CHECK-NOT: Interpolate at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @InterpolateLinearAsymmetric(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x12x12xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 12]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x12x12xf16>

    return %0 : tensor<1x3x12x12xf16>

    // CHECK:       Interpolate at 'loc({{[^']+}})' can be optimized using SEP
    // CHECK-NEXT:    Case is already supported
}

// -----

func.func @InterpolateLinearAsymmetricLargeScale(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x72x72xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
                               mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [12.000000e+00, 12.000000e+00], sizes_attr = [72, 72]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x72x72xf16>

    return %0 : tensor<1x3x72x72xf16>

    // CHECK-NOT: Interpolate at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @InterpolateLinearPytorchHalfPixel(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x12x12xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64,
                               mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 12]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x12x12xf16>

    return %0 : tensor<1x3x12x12xf16>

    // CHECK:       Interpolate at 'loc({{[^']+}})' might potentially be optimized using SEP
    // CHECK-NEXT:    Case might have more constraints, as the proposal needs further analysis
}

// -----

func.func @InterpolateLinearAlignCorners(%arg0: tensor<1x3x6x6xf16>) -> tensor<1x3x12x12xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64,
                               mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
                               pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 12]
    } : tensor<1x3x6x6xf16> -> tensor<1x3x12x12xf16>

    return %0 : tensor<1x3x12x12xf16>

    // CHECK-NOT: Interpolate at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @TransposedConvolution(%arg0: tensor<1x32x30x30xf16>) -> tensor<1x16x60x60xf16> {
    %cst = const.Declare tensor<16x32x2x2xf16> = dense<1.000000e+00> : tensor<16x32x2x2xf16>
    %0 = IE.TransposedConvolution(%arg0, %cst) {
        strides = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, output_padding = [0, 0]
    } : tensor<1x32x30x30xf16>, tensor<16x32x2x2xf16> -> tensor<1x16x60x60xf16>
    return %0 : tensor<1x16x60x60xf16>

    // CHECK: TransposedConvolution at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @DilatedConvolution(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16> {
    %cst = const.Declare tensor<16x16x3x3xf32> = dense<1.000000e+00> : tensor<16x16x3x3xf32>
    %0 = IE.Convolution(%arg0, %cst) {
            strides = [1, 1], pads_begin = [2, 2], pads_end = [2, 2], dilations = [2, 2]
    } : tensor<1x16x32x32xf16>, tensor<16x16x3x3xf32> -> tensor<1x16x32x32xf16>
    return %0 : tensor<1x16x32x32xf16>

    // CHECK: Dilated convolution at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @PadConstantZero(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x35x35xf16> {
    %0 = IE.Pad(%arg0) {
        mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 2, 2]
    } : tensor<1x16x32x32xf16> -> tensor<1x16x35x35xf16>
    return %0 : tensor<1x16x35x35xf16>

    // CHECK: Pad operation at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @PadConstantNonZero(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x35x35xf16> {
    %0 = IE.Pad(%arg0) {
        mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 1.000000e+00 : f64, pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 2, 2]
    } : tensor<1x16x32x32xf16> -> tensor<1x16x35x35xf16>
    return %0 : tensor<1x16x35x35xf16>

    // CHECK:       Pad operation at 'loc({{[^']+}})' can be optimized using SEP
    // CHECK-NEXT:    Would require an extra small DMA to bring the constant '1.00' to CMX, which could hurt performance
}

// -----

func.func @PadConstantZeroNonSpatial(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x18x32x32xf16> {
    %0 = IE.Pad(%arg0) {
        mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 1, 0, 0], pads_end_attr = [0, 1, 0, 0]
    } : tensor<1x16x32x32xf16> -> tensor<1x18x32x32xf16>
    return %0 : tensor<1x18x32x32xf16>

    // CHECK-NOT: Pad operation at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @PadNonConstant(%arg0: tensor<1x16x32x32xf16>) -> tensor<1x16x34x34xf16> {
    %0 = IE.Pad(%arg0) {
        mode = #IE.pad_mode<EDGE>, pads_begin_attr = [0, 0, 1, 1], pads_end_attr = [0, 0, 1, 1]
    } : tensor<1x16x32x32xf16> -> tensor<1x16x34x34xf16>
    return %0 : tensor<1x16x34x34xf16>

    // CHECK: Pad operation at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @Tile3DSpatial(%arg0: tensor<16x5x5xf16>) -> tensor<16x10x15xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 3]} : tensor<16x5x5xf16> -> tensor<16x10x15xf16>
    return %0 : tensor<16x10x15xf16>

    // CHECK: Tile operation at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @Tile4DSpatial(%arg0: tensor<1x16x5x5xf16>) -> tensor<1x16x10x15xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 1, 2, 3]} : tensor<1x16x5x5xf16> -> tensor<1x16x10x15xf16>
    return %0 : tensor<1x16x10x15xf16>

    // CHECK: Tile operation at 'loc({{[^']+}})' can be optimized using SEP
}

// -----

func.func @TileChannels(%arg0: tensor<1x16x5x5xf16>) -> tensor<1x32x5x5xf16> {
    %0 = IE.Tile(%arg0) {repeats_values = [1, 2, 1, 1]} : tensor<1x16x5x5xf16> -> tensor<1x32x5x5xf16>
    return %0 : tensor<1x32x5x5xf16>

    // CHECK: Tile operation at 'loc({{[^']+}})' can be optimized using SEP
}
