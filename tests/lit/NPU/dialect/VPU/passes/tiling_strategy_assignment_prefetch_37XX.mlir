//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment %s | FileCheck %s
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
        ppe = #VPU.PPEStub<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
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
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x244x244xf16> -> tensor<1x16x244x244xf16>
    return %1 : tensor<1x16x244x244xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      : tensor<1x16x244x244xf16> -> tensor<1x16x244x244xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x244xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSoftMaxWithSoK
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x4096x4096xf16>
func.func @SplitSoftMaxWithSoK(%arg0: tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>
    return %0 : tensor<1x8x4096x4096xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 137, 1]}
    // CHECK-SAME:      : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x4096x4096xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSoftMaxOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func.func @SplitSoftMaxOverW(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]}
    // CHECK-SAME:      : tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1NormalizeSplitOverH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x512x1x32xf16, {order = #NHWC}>
func.func @MVN1NormalizeSplitOverH(%arg0: tensor<1x512x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x512x1x32xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x512x256x256xf16, {order = #NHWC}>, tensor<1x512x1x32xf16, {order = #NHWC}> -> tensor<1x512x256x256xf16, {order = #NHWC}>
    return %0 :  tensor<1x512x256x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
    // CHECK-SAME:          tilingStrategy = [1, 1, 255, 1]
    // CHECK:       return [[OUTPUT]] : tensor<1x512x256x256xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
        tensor<1x32x64x64xf16> -> tensor<1x32x256x256xf16>

    return %0 : tensor<1x32x256x256xf16>

    // CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]]
    // CHECK-SAME:      tilingStrategy = [1, 1, 3, 1]
    // CHECK-SAME:      : tensor<1x32x64x64xf16>
    // CHECK-SAME:      -> tensor<1x32x256x256xf16>

    // CHECK:       return [[INTERP0]] : tensor<1x32x256x256xf16>
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

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 5]
    // CHECK-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x35x35xf16, {order = #NHWC}>
func.func @InterpSplitOverHW(
    %input1: tensor<1x128x35x35xf16, {order = #NHWC}>)
            -> tensor<1x128x168x335xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%input1) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [168, 335]} :
        tensor<1x128x35x35xf16, {order = #NHWC}> -> tensor<1x128x168x335xf16, {order = #NHWC}>
    return %0 : tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 8]
    // CHECK-SAME:  : tensor<1x128x35x35xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x128x168x335xf16, {order = #NHWC}>

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

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]])
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:  : tensor<1x64x31x31xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x121x121xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @SplitPReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitPReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x1280xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

    // CHECK:       [[OUTPUT:%.+]] = VPU.PRelu([[INPUT]], [[CST]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: func.func @SplitLeakyReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.LeakyRelu([[INPUT]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
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
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [576, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}>, tensor<576x1x1x4xsi32, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:     rawFilterShape = [144, 144, 3, 3]
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]])
    // CHECK-SAME:          op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:     pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      rawFilterShape = [576, 144, 3, 3]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 5, 1]
    // CHECK-SAME:          -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x340x340xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:      tilingStrategy = [1, 1, 7, 1]
    // CHECK-SAME:      } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x340xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x100x100xf16, {order = #NHWC}>, tensor<128x32x3x3xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
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
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } : tensor<1x16x210x210xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x210x210xf16, {order = #NHWC}>

    return %0 : tensor<1x32x210x210xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x210x210xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x32x210x210xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<240x32x7x7xf16, {order = #NHWC}>, tensor<240x1x1x4xsi32> -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-NOT:           tilingStrategy
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
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [768, 32, 7, 7],
        strides = [1, 1]
    } : tensor<1x32x30x30xf16, {order = #NHWC}>, tensor<768x32x7x7xf16, {order = #NHWC}>, tensor<768x1x1x4xsi32> -> tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<768x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
    // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]}
    // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}>, tensor<512x1x1x4xsi32> -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<512x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [512, 256, 3, 3]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:          -> tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x512x14x14xf16, {order = #NHWC}>
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

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[ELTWISE_0]] : tensor<1x512x28x28xf16, {order = #NHWC}>
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
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x70x70xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x70x70xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    return %1 : tensor<1x64x70x70xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]])
    // CHECK-SAME:              op_type = #VPU.eltwise_type<ADD>
    // CHECK-NOT:               tilingStrategy
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
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x80x80xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>, tensor<160x1x1x4xsi32, {order = #NCHW}> -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
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

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 4]
    // CHECK-SAME:      -> tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x11520xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16>) -> tensor<1x1x1x368640xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x368640xf16> -> tensor<1x1x1x368640xf16>

    return %0 : tensor<1x1x1x368640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x8x10001xf16, {order = #NHWC}>
func.func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x8x10001xf16, {order = #NHWC}>) -> tensor<1x16x8x10001xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    } -> tensor<1x16x8x10001xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x10001xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[INPUT]])
    // CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x8x10001xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL_0]] : tensor<1x16x8x10001xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PrefetchingNCE
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x80x80xf16>
func.func @PrefetchingNCE(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x48x80x80xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<48x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %0 = VPU.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        } : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        rawFilterShape = [48, 64, 5, 5],
        strides = [1, 1]
    } : tensor<1x64x80x80xf16, {order = #NHWC}>, tensor<48x64x5x5xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x80x80xf16, {order = #NHWC}>

    return %1 : tensor<1x48x80x80xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<48x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<48x1x1x4xsi32>

    // CHECK:       [[MEMPERMUTE:%.+]] = VPU.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      rawFilterShape = [48, 64, 5, 5]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x48x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[CONV_0]]
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
    // CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @LogSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Log(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Log([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @AbsSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @AbsSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Abs(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Abs([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitFloorModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.FloorMod([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Mod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Mod([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitPowerEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Power([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitLogicalOrEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalOr([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitLogicalXorEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalXor([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8>
func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
    %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
    return %0 : tensor<1x10x256x256xi8>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>
}

// -----

// CHECK-LABEL: @SplitNotEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8>
func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
    %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
    return %0 : tensor<1x10x256x256xi8>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>
}

// -----

// CHECK-LABEL: @SplitSoftPlusActivationSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitSoftPlusActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.SoftPlus(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.SoftPlus([[INPUT_0]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitLessEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8>
func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
    %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
    return %0 : tensor<1x10x256x256xi8>
    // CHECK:       [[OUTPUT:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>
}

// -----

// CHECK-LABEL: @SplitLessEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8>
func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
    %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
    return %0 : tensor<1x10x256x256xi8>
    // CHECK:       [[OUTPUT:%.+]] = VPU.LessEqual([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>
}

// -----

// CHECK-LABEL: @SplitGreaterEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8>
func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xi8> {
    %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>
    return %0 : tensor<1x10x256x256xi8>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xi8>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xi8>
}

// -----

// CHECK-LABEL: @SplitGreaterEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GreaterEqual([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitErfOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitErfOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Erf(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Erf([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitFloorOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitFloorOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Floor(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Floor([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @TanSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Tan(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Tan([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SwishSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Swish([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @HSigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.HSigmoid([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitNegativeActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Negative(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Negative([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitCeilingActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Ceiling([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitSignActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Sign(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sign([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}


// -----

// CHECK-LABEL: @SplitSelectEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>, %arg2: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Select([[INPUT_0]], [[INPUT_1]], [[INPUT_2]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 5, 1, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitAddEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitAddEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Round([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitGeluActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Gelu([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitReduceSum
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16>
func.func @SplitReduceSum(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.ReduceSum(%arg0) {axes_value = [1], keep_dims} : tensor<1x12x368x480xf16> -> tensor<1x1x368x480xf16>
    return %0 : tensor<1x1x368x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReduceSum([[INPUT]]) {
    // CHECK-SAME:  axes_value = [1], keep_dims, tilingStrategy = [1, 1, 1, 3]} : tensor<1x12x368x480xf16> -> tensor<1x1x368x480xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x368x480xf16>
}

// -----

// CHECK-LABEL: @SplitTopK
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>)
func.func @SplitTopK(%arg0: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    %aux = const.Declare tensor<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %aux) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x512x512xf16>, tensor<1x1x512x512xsi32>
    %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    return %0, %target_shape : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>

    // CHECK: [[AUX:%.+]] = const.Declare tensor<1x1x1x80xui8> = dense<0> : tensor<1x1x1x80xui8>
    // CHECK: [[OUTPUT_VALUE:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[INPUT_0]], [[AUX]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>, tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x512x512xf16>, tensor<1x1x1x80xui8> -> tensor<1x1x512x512xf16>, tensor<1x1x512x512xsi32>
    // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>
}

// -----

// CHECK-LABEL: func.func @SplitStridedSliceOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x640xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 1280], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x640xf16>
    return %0 : tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.StridedSlice([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x640xf16>
}

// -----

// CHECK-LABEL: @SplitLogicalNotEltwiseSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalNot([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOCInputOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x32x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOCInputOversize(%arg0: tensor<1x512x32x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 512, 3, 3],
        strides = [1, 1]
    } : tensor<1x512x32x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x32x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK-DAG:         [[WEIGHTS:%.+]] = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:         [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:                tilingStrategy = [1, 2, 5, 1]
    // CHECK-SAME:        -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:         return [[CONV]] : tensor<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOHFilterOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOHFilterOversize(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 512, 3, 3],
        strides = [1, 1]
    } : tensor<1x512x64x64xf16, {order = #NHWC}>, tensor<256x512x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:         [[WEIGHTS:%.+]] = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:         [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    // CHECK:         [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:                tilingStrategy = [1, 2, 9, 1]
    // CHECK-SAME:        -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:         return [[CONV]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitSquaredDiffEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitSquaredDiffEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.SquaredDiff(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.SquaredDiff([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

// CHECK-LABEL: @ReLUSplitOverC
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
func.func @ReLUSplitOverC(%arg0: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
    %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16> -> tensor<1x80x80x80xf16>
    return %0 : tensor<1x80x80x80xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]} : tensor<1x80x80x80xf16> -> tensor<1x80x80x80xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReLUSplitOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}>
func.func @ReLUSplitOverH(%arg0: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}> {
    %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>
    return %0 : tensor<1x80x80x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x80x80x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCECompressConv
// CHECK-SAME:        [[ARG_0:%[^:]+]]: tensor<1x4x448x224xf16, {order = #NHWC}>,
// CHECK-SAME:        [[ARG_1:%[^:]+]]: tensor<64x1x1x160xf16, {order = #NHWC}>,
// CHECK-SAME:        [[ARG_2:%[^:]+]]: tensor<64x1x1x4xsi32>) -> tensor<1x64x224x112xf16, {order = #NHWC}>
func.func @SplitNCECompressConv(
        %arg0: tensor<1x4x448x224xf16, {order = #NHWC}>,
        %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>,
        %arg2: tensor<64x1x1x4xsi32>)
        -> tensor<1x64x224x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
        ppe = #VPU.PPEStub<>,
        cm_sp_pattern = 15 : i64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        rawFilterShape = [64, 4, 7, 7], strides = [2, 2],
        pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>
    } -> tensor<1x64x224x112xf16, {order = #NHWC}>

    return %0 : tensor<1x64x224x112xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.CompressConvolution([[ARG_0]], [[ARG_1]], [[ARG_2]]) {
    // CHECK-SAME:      cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
    // CHECK-SAME:      rawFilterShape = [64, 4, 7, 7], strides = [2, 2], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:      -> tensor<1x64x224x112xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x224x112xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitDFT
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16>
func.func @SplitDFT(%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.DFT(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
    return %0 : tensor<1x120x64x64x2xf16>

// CHECK: [[CST:%.+]]  = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
// CHECK: [[OUTPUT:%.+]] = VPU.DFT([[ARG_0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1], tilingStrategy = [1, 3, 1, 1, 1]} : tensor<1x120x64x64x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
// CHECK: return [[OUTPUT]] : tensor<1x120x64x64x2xf16>

}

// -----

// CHECK-LABEL: @SplitIDFT
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16>
func.func @SplitIDFT(%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16> {
     %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.IDFT(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
    return %0 : tensor<1x120x64x64x2xf16>

// CHECK: [[CST:%.+]]  = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
// CHECK: [[OUTPUT:%.+]] = VPU.IDFT([[ARG_0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1], tilingStrategy = [1, 3, 1, 1, 1]} : tensor<1x120x64x64x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
// CHECK: return [[OUTPUT]] : tensor<1x120x64x64x2xf16>
}

// -----

// CHECK-LABEL: @SplitRDFTUncut
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x120x64x64xf16>) -> tensor<1x120x64x64x2xf16>
func.func @SplitRDFTUncut(%arg0: tensor<1x120x64x64xf16>) -> tensor<1x120x64x64x2xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.RDFTUncut(%arg0, %cst) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
    return %0 : tensor<1x120x64x64x2xf16>

// CHECK: [[CST:%.+]]  = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
// CHECK: [[OUTPUT:%.+]] = VPU.RDFTUncut([[ARG_0]], [[CST]]) {axes_attr = [2, 3], signal_size_attr = [-1, -1], tilingStrategy = [1, 2, 1, 1, 1]} : tensor<1x120x64x64xf16>, tensor<16384xf16> -> tensor<1x120x64x64x2xf16>
// CHECK: return [[OUTPUT]] : tensor<1x120x64x64x2xf16>
}

// -----

// CHECK-LABEL: @SplitIRDFTLastAxis
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16>
func.func @SplitIRDFTLastAxis(%arg0: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16> {
    %cst = const.Declare tensor<16384xf16> = dense<2.0> : tensor<16384xf16>
    %0 = VPU.IRDFTLastAxis(%arg0, %cst) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x120x64x33x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64xf16>
    return %0 : tensor<1x120x64x64xf16>

// CHECK: [[CST:%.+]]  = const.Declare tensor<16384xf16> = dense<2.000000e+00> : tensor<16384xf16>
// CHECK: [[OUTPUT:%.+]] = VPU.IRDFTLastAxis([[ARG_0]], [[CST]]) {axes_attr = [3], signal_size_attr = [-1], tilingStrategy = [1, 2, 1, 1]} : tensor<1x120x64x33x2xf16>, tensor<16384xf16> -> tensor<1x120x64x64xf16>
// CHECK: return [[OUTPUT]] : tensor<1x120x64x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.0>

// CHECK-LABEL: func.func @PrefetchTilingWithParentConsidered
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<4096x512x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<512x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<4096x1x1x4xsi32, {order = #NHWC}>
func.func @PrefetchTilingWithParentConsidered(
        %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
        %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
        %weights2: tensor<4096x512x1x1x!qElemType, {order = #NHWC}>,
        %weights_table1: tensor<512x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<4096x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x4096x7x7x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
            : tensor<1x512x14x14x!qElemType, {order = #NHWC}>, tensor<512x512x3x3x!qElemType, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NHWC}> -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights2, %weights_table2) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [4096, 512, 1, 1], strides = [1, 1]}
            : tensor<1x512x7x7x!qElemType, {order = #NHWC}>, tensor<4096x512x1x1x!qElemType, {order = #NHWC}>, tensor<4096x1x1x4xsi32, {order = #NHWC}> -> tensor<1x4096x7x7x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x4096x7x7x!qElemType, {order = #NHWC}>

    // Prefetching mode is triggered for the child conv
    // with tiled parent memory considered
    // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:  tilingStrategy = [1, 4, 1, 1]
    // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:  tilingStrategy = [1, 4, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.0>

// CHECK-LABEL: func.func @PrefetchTilingWithSOKParentConsidered
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x1024x3x3x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<4096x512x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<512x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<4096x1x1x4xsi32, {order = #NHWC}>
func.func @PrefetchTilingWithSOKParentConsidered(
        %input: tensor<1x1024x14x14x!qElemType, {order = #NHWC}>,
        %weights1: tensor<512x1024x3x3x!qElemType, {order = #NHWC}>,
        %weights2: tensor<4096x512x1x1x!qElemType, {order = #NHWC}>,
        %weights_table1: tensor<512x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<4096x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x4096x7x7x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
        rawFilterShape = [512, 1024, 3, 3], strides = [2, 2]}
            : tensor<1x1024x14x14x!qElemType, {order = #NHWC}>, tensor<512x1024x3x3x!qElemType, {order = #NHWC}>, tensor<512x1x1x4xsi32, {order = #NHWC}> -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights2, %weights_table2) {
        ppe = #VPU.PPEStub<>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [4096, 512, 1, 1], strides = [1, 1]}
            : tensor<1x512x7x7x!qElemType, {order = #NHWC}>, tensor<4096x512x1x1x!qElemType, {order = #NHWC}>, tensor<4096x1x1x4xsi32, {order = #NHWC}> -> tensor<1x4096x7x7x!qElemType, {order = #NHWC}>
    return %1 : tensor<1x4096x7x7x!qElemType, {order = #NHWC}>

    // Prefetching mode is triggered for the child conv
    // with tiled parent memory considered
    // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:  tilingStrategy = [1, 4, 1, 1]
    // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PermuteOpWithHugeInputSize
// CHECK-SAME:        [[ARG_0:%[^:]+]]: tensor<1x4x2048x336xf16>
func.func @PermuteOpWithHugeInputSize(%arg0: tensor<1x4x2048x336xf16>) -> tensor<1x4x2048x336xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 4 : i64,
                                 multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                                 ppe = #VPU.PPEStub<>} -> tensor<1x4x2048x336xf16, {order = #NHWC}>
    return %0 : tensor<1x4x2048x336xf16, {order = #NHWC}>


    //CHECK: [[RET:%.+]] = VPU.NCE.Permute([[ARG_0]]) {
    //CHECK-SAME{LITERAL}:            tilingStrategy = [1, 1, 3, 1]
    //CHECK:                  tensor<1x4x2048x336xf16, {order = #NHWC}>

    //CHECK: return [[RET]] : tensor<1x4x2048x336xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @PReLUSOH_TileOverH
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x256x64x64xf16>)
func.func @PReLUSOH_TileOverH(%arg0: tensor<1x256x64x64xf16>) -> tensor<1x256x64x64xf16> {
    %cst = const.Declare tensor<1x256x1x1xf16> = dense<-1.000000e+01> : tensor<1x256x1x1xf16>
    %0 = VPU.PRelu(%arg0, %cst) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
        : tensor<1x256x64x64xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %0 : tensor<1x256x64x64xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x256x1x1xf16> = dense<-1.000000e+01> : tensor<1x256x1x1xf16>
    // CHECK:      [[OUTPUT:%.+]] = VPU.PRelu([[INPUT]], [[CST]]) {
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:    tilingStrategy = [1, 1, 2, 1]

    // CHECK:      return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @RMSSOK_TileOverC
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x192x192x512xf16>)
func.func @RMSSOK_TileOverC(%arg0: tensor<1x192x192x512xf16>) -> tensor<1x192x192x512xf16> {
    %cst = const.Declare tensor<1x1x1x512xf16> = dense<1.0> : tensor<1x1x1x512xf16>
    %0 = VPU.RMS(%arg0, %cst) {
        eps = 9.9999999747524271E-7 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
        : tensor<1x192x192x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x192x192x512xf16>
    return %0 : tensor<1x192x192x512xf16>

    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+00> : tensor<1x1x1x512xf16>
    // CHECK:      [[OUTPUT:%.+]] = VPU.RMS([[INPUT]], [[CST]]) {
    // CHECK-SAME:    eps = 9.9999999747524271E-7 : f64
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:    tilingStrategy = [1, 24, 1, 1]

    // CHECK:      return [[OUTPUT]] : tensor<1x192x192x512xf16>
}
