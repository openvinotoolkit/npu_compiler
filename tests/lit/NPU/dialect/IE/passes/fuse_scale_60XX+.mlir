//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-scale --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU6010


!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesToConvolution
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseChannelWiseScalesToConvolution(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}


// -----

!quantileType = !quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK-LABEL: @FuseChannelWiseScalesWithAffineReshapeTranspose
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x16384x4096x!quantileType>, [[ARG1:%.+]]: tensor<1x1x1024x4096xf16>, [[ARG2:%.+]]: tensor<16384x1xf16>)

func.func @FuseChannelWiseScalesWithAffineReshapeTranspose(%arg0: tensor<1x1x16384x4096x!quantileType>,
                                                           %arg1: tensor<1x1x1024x4096xf16>,
                                                           %arg2: tensor<16384x1xf16>) -> tensor<1x16384x1024x1xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x16384x4096x!quantileType> -> tensor<1x1x16384x4096xf16>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1024, 4096, 1, 1]} : tensor<1x1x1024x4096xf16> -> tensor<1024x4096x1x1xf16>
    %2 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [16384, 4096, 1, 1]} : tensor<1x1x16384x4096xf16> -> tensor<16384x4096x1x1xf16>
    %3 = IE.Convolution(%1, %2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x4096x1x1xf16>, tensor<16384x4096x1x1xf16> -> tensor<1024x16384x1x1xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1024, 1, 16384]} : tensor<1024x16384x1x1xf16> -> tensor<1x1024x1x16384xf16>
    %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1024x1x16384xf16> -> tensor<1x16384x1024x1xf16>
    %6 = IE.AffineReshape(%arg2) {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 16384, 1, 1]} : tensor<16384x1xf16> -> tensor<1x16384x1x1xf16>
    %7 = IE.Multiply(%5, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16384x1024x1xf16>, tensor<1x16384x1x1xf16> -> tensor<1x16384x1024x1xf16>

    return %7 : tensor<1x16384x1024x1xf16>

    // CHECK: [[DQ:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x16384x4096x!quantileType> -> tensor<1x1x16384x4096xf16>
    // CHECK: [[RESHAPE_IN:%.+]] = IE.AffineReshape([[ARG1]])
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[RESHAPE_S:%.+]] = IE.Reshape([[ARG2]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[RESHAPE_S]])
    // CHECK-SAME: tensor<1x256x8x8xf16> -> tensor<1x256x8x8xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[RESHAPE_IN]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1024x4096x1x1xf16>, tensor<16384x4096x1x1xf16>, tensor<16384x1x1x1xf32> -> tensor<1024x16384x1x1xf16>
    // CHECK: [[RESHAPE_OUT:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE_OUT]])
    // CHECK: return [[TRANSPOSE]] : tensor<1x16384x1024x1xf16>
}


// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesToConvolutionNF4
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseChannelWiseScalesToConvolutionNF4(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}


// -----

!quantileType = !quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>

// CHECK-LABEL: @FuseChannelWiseScalesToConvolutionQuantileFloat
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!quantileType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseChannelWiseScalesToConvolutionQuantileFloat(%arg0: tensor<1x1x128x1024x!quantileType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!quantileType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[DQ:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!quantileType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 2.000000e+00>

// CHECK-LABEL: @FuseForNonOneQuantScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseForNonOneQuantScale(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}


// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @NotFuseForMismatchedMultiplyShape
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x2x2xf16>, [[ARG2:%.+]]: tensor<1x128x2x1xf16>)

func.func @NotFuseForMismatchedMultiplyShape(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x2x2xf16>, %arg2: tensor<1x128x2x1xf16>)
        -> tensor<1x128x2x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x2x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x2x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x2x2xf16>, tensor<1x128x2x1xf16> -> tensor<1x128x2x2xf16>

    return %3 : tensor<1x128x2x2xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: return [[MULTIPLY]] : tensor<1x128x2x2xf16>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @FuseForStaticScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseForStaticScale(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], static_scale = 8.400000e+01 : f32, strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}


// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @NotFuseForPostOp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @NotFuseForPostOp(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>, strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: return [[MULTIPLY]] : tensor<1x128x4x2xf16>
}


// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @NotFuseForClamp
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)
func.func @NotFuseForClamp(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: return [[MULTIPLY]] : tensor<1x128x4x2xf16>
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, clamp = {min = 0.000000e+00 : f64, max = 6.000000e+00 : f64}} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>
}

// -----

!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>:f16, 1.000000e+00>

// CHECK-LABEL: @FuseIntoAlreadyHaveScale
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>, [[ARG3:%.+]]: tensor<1x128x1x1xf16>, [[ARG4:%.+]]: tensor<128x1x1x1xf32>)

func.func @FuseIntoAlreadyHaveScale(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>, %arg3: tensor<1x128x1x1xf16>, %arg4: tensor<128x1x1x1xf32>)
        -> tensor<1x128x4x2xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1, %arg3, %arg4) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<1x128x1x1xf16> , tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[MERGED_SCALE:%.+]] = IE.Multiply([[ARG4]], [[RESHAPE_F32]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x1x1x1xf32>, tensor<128x1x1x1xf32> -> tensor<128x1x1x1xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], {{%.+}}, [[ARG3]], [[MERGED_SCALE]])
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesWithTransposeOnly
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x32x16xf16>, [[ARG2:%.+]]: tensor<1x1x128x1xf16>)
func.func @FuseChannelWiseScalesWithTransposeOnly(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x32x16xf16>,
                                                  %arg2: tensor<1x1x128x1xf16>) -> tensor<1x32x128x16xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x16xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x32x16xf16>
    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x128x32x16xf16> -> tensor<1x32x128x16xf16>
    %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x128x16xf16>, tensor<1x1x128x1xf16> -> tensor<1x32x128x16xf16>
    return %4 : tensor<1x32x128x16xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[RESHAPE_S:%.+]] = IE.AffineReshape([[ARG2]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[RESHAPE_S]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x16xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x32x16xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]])
    // CHECK-SAME: : tensor<1x128x32x16xf16> -> tensor<1x32x128x16xf16>
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[TRANSPOSE]] : tensor<1x32x128x16xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesWithAffineReshapeOnly
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x32x1xf16>, [[ARG2:%.+]]: tensor<1x1x128x1x1xf16>)
func.func @FuseChannelWiseScalesWithAffineReshapeOnly(%arg0: tensor<1x1x128x1024x!qElemType>,
                                                      %arg1: tensor<1x1024x32x1xf16>,
                                                      %arg2: tensor<1x1x128x1x1xf16>) -> tensor<1x1x128x4x8xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x1xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x32x1xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3, 4], [4]], shape_value = [1, 1, 128, 4, 8]} : tensor<1x128x32x1xf16> -> tensor<1x1x128x4x8xf16>
    %4 = IE.Multiply(%3, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x128x4x8xf16>, tensor<1x1x128x1x1xf16> -> tensor<1x1x128x4x8xf16>
    return %4 : tensor<1x1x128x4x8xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[RESHAPE_S:%.+]] = IE.AffineReshape([[ARG2]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[RESHAPE_S]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x1xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x32x1xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[CONV]])
    // CHECK-SAME: : tensor<1x128x32x1xf16> -> tensor<1x1x128x4x8xf16>
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[RESHAPE]] : tensor<1x1x128x4x8xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesWithTransposeAffineReshape
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x8xf16>, [[ARG2:%.+]]: tensor<1x1x128x1x1xf16>)
func.func @FuseChannelWiseScalesWithTransposeAffineReshape(%arg0: tensor<1x1x128x1024x!qElemType>,
                                                           %arg1: tensor<1x1024x4x8xf16>,
                                                           %arg2: tensor<1x1x128x1x1xf16>) -> tensor<1x4x128x2x4xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x8xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x8xf16>
    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x128x4x8xf16> -> tensor<1x4x128x8xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 4, 128, 2, 4]} : tensor<1x4x128x8xf16> -> tensor<1x4x128x2x4xf16>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x128x2x4xf16>, tensor<1x1x128x1x1xf16> -> tensor<1x4x128x2x4xf16>
    return %5 : tensor<1x4x128x2x4xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[RESHAPE_S:%.+]] = IE.AffineReshape([[ARG2]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[RESHAPE_S]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[ARG1]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x8xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x8xf16>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]])
    // CHECK-SAME: : tensor<1x128x4x8xf16> -> tensor<1x4x128x8xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME: : tensor<1x4x128x8xf16> -> tensor<1x4x128x2x4xf16>
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[RESHAPE]] : tensor<1x4x128x2x4xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NotFuseForAffineReshapeSplitAxis
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)
func.func @NotFuseForAffineReshapeSplitAxis(%arg0: tensor<1x1x128x1024x!qElemType>,
                                                   %arg1: tensor<1x1024x4x2xf16>,
                                                   %arg2: tensor<1x128x1x1xf16>) -> tensor<1x2x64x8xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.AffineReshape(%2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 64, 8]} : tensor<1x128x4x2xf16> -> tensor<1x2x64x8xf16>
    %4 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1, 2], [3], [3]], shape_value = [1, 2, 64, 1]} : tensor<1x128x1x1xf16> -> tensor<1x2x64x1xf16>
    %5 = IE.Multiply(%3, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x64x8xf16>, tensor<1x2x64x1xf16> -> tensor<1x2x64x8xf16>
    return %5 : tensor<1x2x64x8xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: return [[MULTIPLY]] : tensor<1x2x64x8xf16>
}


// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NotFuseForDifferentMultiplyAxis
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x1x1xf16>, [[ARG2:%.+]]: tensor<1x128xf16>)
func.func @NotFuseForDifferentMultiplyAxis(%arg0: tensor<1x1x128x1024x!qElemType>,
                                                           %arg1: tensor<1x1024x1x1xf16>,
                                                           %arg2: tensor<1x128xf16>) -> tensor<128x128xf16> {
    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x1x1xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x1x1xf16>
    %3 = IE.Transpose(%2) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x128x1x1xf16> -> tensor<1x1x128x1xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [0], [0, 1]], shape_value = [128, 1]} : tensor<1x1x128x1xf16> -> tensor<128x1xf16>
    %5 = IE.Multiply(%4, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<128x1xf16>, tensor<1x128xf16> -> tensor<128x128xf16>
    return %5 : tensor<128x128xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: return [[MULTIPLY]] : tensor<128x128xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NotFuseForFakeQuantizeUser
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @NotFuseForFakeQuantizeUser(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %high = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%arg1, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    %4 = IE.FakeQuantize(%3, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x128x4x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x4x2xf16>


    return %4 : tensor<1x128x4x2xf16>

    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize
    // CHECK: return [[FAKEQUANTIZE]] : tensor<1x128x4x2xf16>
}


// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseForFakeQuantizeInput
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x1024x4x2xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @FuseForFakeQuantizeInput(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x1024x4x2xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x4x2xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %high = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %fq = IE.FakeQuantize(%arg1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x1024x4x2xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1024x4x2xf16>

    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%fq, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x4x2xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x4x2xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x4x2xf16>

    return %3 : tensor<1x128x4x2xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG1]]
    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ]], [[RESHAPE_W]], [[RESHAPE_F32]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x4x2xf16>, tensor<128x1024x1x1xf16>, tensor<128x1x1x1xf32> -> tensor<1x128x4x2xf16>
    // CHECK: return [[CONV]] : tensor<1x128x4x2xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @NotFuseForFakeQuantizeInputWithTranspose
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x128x1024x!qElemType>, [[ARG1:%.+]]: tensor<1x32x1024x1xf16>, [[ARG2:%.+]]: tensor<1x128x1x1xf16>)

func.func @NotFuseForFakeQuantizeInputWithTranspose(%arg0: tensor<1x1x128x1024x!qElemType>, %arg1: tensor<1x32x1024x1xf16>, %arg2: tensor<1x128x1x1xf16>)
        -> tensor<1x128x32x1xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %high = const.Declare tensor<1x1x1x1xf16> = dense<57.1374702> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %fq = IE.FakeQuantize(%arg1, %low, %high, %low, %high) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 256 : i64} : tensor<1x32x1024x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x1xf16>
    %transposed_input = IE.Transpose(%fq) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x32x1024x1xf16> -> tensor<1x1024x32x1xf16>

    %0 = IE.Dequantize(%arg0) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [128, 1024, 1, 1]} : tensor<1x1x128x1024xf16> -> tensor<128x1024x1x1xf16>
    %2 = IE.Convolution(%transposed_input, %1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1024x32x1xf16>, tensor<128x1024x1x1xf16> -> tensor<1x128x32x1xf16>
    %3 = IE.Multiply(%2, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x128x32x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x32x1xf16>

    return %3 : tensor<1x128x32x1xf16>

    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[ARG1]]
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[FQ]]) {order_value = #NHCW} : tensor<1x32x1024x1xf16> -> tensor<1x1024x32x1xf16>

    // CHECK: [[DQ:%.+]] = IE.Dequantize([[ARG0]]) {dstElemType = f16} : tensor<1x1x128x1024x!qElemType> -> tensor<1x1x128x1024xf16>
    // CHECK: [[RESHAPE_W:%.+]] = IE.AffineReshape([[DQ]])
    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG2]])
    // CHECK-SAME: tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_F32:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.Convolution([[TRANSPOSE]], [[RESHAPE_W]], [[RESHAPE_F32]]
    // CHECK: return [[CONV]] : tensor<1x128x32x1xf16>
}

// -----

!qElemType = !quant.uniform<u8:f32, 1.000000e+00>

// CHECK-LABEL: @FuseChannelWiseScalesWithTransposeAffineReshapeMaxPool
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x128x4x8xf16>, [[ARG1:%.+]]: tensor<1x128x1x1xf16>)
func.func @FuseChannelWiseScalesWithTransposeAffineReshapeMaxPool(%arg0: tensor<1x128x4x8xf16>,
                                                           %arg1: tensor<1x128x1x1xf16>) -> tensor<4x128x2x4xf16> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            clamp = {min = 0.000000e+00 : f64, max = 6.000000e+00 : f64}
        } : tensor<1x128x4x8xf16> -> tensor<1x128x4x8xf16>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>} : tensor<1x128x4x8xf16> -> tensor<1x4x128x8xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [4, 128, 2, 4]} : tensor<1x4x128x8xf16> -> tensor<4x128x2x4xf16>
    %3 = IE.Multiply(%2, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x128x2x4xf16>, tensor<1x128x1x1xf16> -> tensor<4x128x2x4xf16>
    return %3 : tensor<4x128x2x4xf16>

    // CHECK: [[SCALE_F32:%.+]] = IE.MaxPool([[ARG1]])
    // CHECK-SAME: : tensor<1x128x1x1xf16> -> tensor<1x128x1x1xf32>
    // CHECK: [[RESHAPE_OUT_S:%.+]] = IE.AffineReshape([[SCALE_F32]])
    // CHECK: [[CONV:%.+]] = IE.MaxPool([[ARG0]], [[RESHAPE_OUT_S]])
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONV]])
    // CHECK-SAME: : tensor<1x128x4x8xf16> -> tensor<1x4x128x8xf16>
    // CHECK: [[RESHAPE:%.+]] = IE.AffineReshape([[TRANSPOSE]])
    // CHECK-SAME: : tensor<1x4x128x8xf16> -> tensor<4x128x2x4xf16>
    // CHECK-NOT: IE.Multiply
    // CHECK: return [[RESHAPE]] : tensor<4x128x2x4xf16>
}

// -----

// CHECK-LABEL: @DoNotFuseChannelWiseScalesNonConstScalar
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x1x4x300xf16>, [[ARG1:%.+]]: tensor<1x1x1x1xf16>)
func.func @DoNotFuseChannelWiseScalesNonConstScalar(
        %arg0: tensor<1x1x4x300xf16>,
        %arg1: tensor<1x1x1x1xf16>) -> tensor<1x300x1x1xf16> {
    %0 = IE.MaxPool(%arg0) {kernel_size = [4, 1], pads_begin = [0, 0], pads_end = [0, 0],
                             rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
            : tensor<1x1x4x300xf16> -> tensor<1x1x1x300xf16>
    %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 300, 1, 1]}
            : tensor<1x1x1x300xf16> -> tensor<1x300x1x1xf16>
    %3 = IE.Multiply(%1, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x300x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x300x1x1xf16>
    return %3 : tensor<1x300x1x1xf16>

    // CHECK: [[POOL0:%.+]] = IE.MaxPool([[ARG0]]) {kernel_size = [4, 1]
    // CHECK-SAME: tensor<1x1x4x300xf16> -> tensor<1x1x1x300xf16>
    // CHECK: [[AR:%.+]] = IE.AffineReshape([[POOL0]])
    // CHECK-SAME: tensor<1x1x1x300xf16> -> tensor<1x300x1x1xf16>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[AR]], [[ARG1]])
    // CHECK-SAME: tensor<1x300x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x300x1x1xf16>
    // CHECK: return [[MUL]] : tensor<1x300x1x1xf16>
}
