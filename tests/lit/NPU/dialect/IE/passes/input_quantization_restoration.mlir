//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --input-quantization-restoration %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX


// CHECK-LABEL: @InputQuantizationRestoration4D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @InputQuantizationRestoration4D(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x800x1280xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.AvgPool(%1) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    return %2 : tensor<1x4x800x1280xf32>

    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[FAKEQUANTIZE]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    // CHECK: return [[AVGPOOL]] : tensor<1x4x800x1280xf32>
}

// -----

// CHECK-LABEL: @InputQuantizationRestoration3D
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x300x560xui8>
func.func @InputQuantizationRestoration3D(%arg0: tensor<1x300x560xui8>) -> tensor<1x300x560xf32> {
    %scale = const.Declare tensor<1x1x1xf32> = dense<0.5> : tensor<1x1x1xf32>
    %add_const = const.Declare tensor<1x1x1xf32> = dense<5.0> : tensor<1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x300x560xui8> -> tensor<1x300x560xf32>
    %1 = IE.Multiply(%0, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x560xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    %2 = IE.Add(%1, %add_const) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x560xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    return %2 : tensor<1x300x560xf32>

    // CHECK-DAG: [[ADD_CONST:%.+]] = const.Declare tensor<1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1xf32> = dense<1.275000e+02> : tensor<1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x300x560xui8> -> tensor<1x300x560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x300x560xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[FAKEQUANTIZE]], [[ADD_CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x300x560xf32>, tensor<1x1x1xf32> -> tensor<1x300x560xf32>
    // CHECK: return [[ADD]] : tensor<1x300x560xf32>
}

// -----

// CHECK-LABEL: @ConvertAsSecondMultiplyInput
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @ConvertAsSecondMultiplyInput(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %weights = const.Declare tensor<4x4x3x3xf32> = dense<1.0> : tensor<4x4x3x3xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%scale, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf32>, tensor<1x4x1600x2560xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Convolution(%1, %weights) {strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %2 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<1.000000e+00> : tensor<4x4x3x3xf32>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FAKEQUANTIZE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationI8
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xi8>
func.func @NoInputQuantizationRestorationI8(%arg0: tensor<1x4x1600x2560xi8>) -> tensor<1x4x1600x2560xf32> {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %weights = const.Declare tensor<4x4x3x3xf32> = dense<1.0> : tensor<4x4x3x3xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xi8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Convolution(%1, %weights) {strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %2 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<1.000000e+00> : tensor<4x4x3x3xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xi8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[MULTIPLY]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationDiffChannel
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @NoInputQuantizationRestorationDiffChannel(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %cst = const.Declare tensor<1x4x1x1xf32> = dense<70.5> : tensor<1x4x1x1xf32>
    %weights = const.Declare tensor<4x4x3x3xf32> = dense<1.0> : tensor<4x4x3x3xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Convolution(%1, %weights) {strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %2 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x4x1x1xf32> = dense<7.050000e+01> : tensor<1x4x1x1xf32>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<1.000000e+00> : tensor<4x4x3x3xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[MULTIPLY]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @InputQuantizationRestorationMultiplyWithSubtract
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @InputQuantizationRestorationMultiplyWithSubtract(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x800x1280xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %zero = const.Declare tensor<1x1x1x1xf32> = dense<5.0> : tensor<1x1x1x1xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %1 = IE.Subtract(%0, %zero) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %3 = IE.AvgPool(%2) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    return %3 : tensor<1x4x800x1280xf32>


    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.0195624866> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.97812432> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[FAKEQUANTIZE]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    // CHECK: return [[AVGPOOL]] : tensor<1x4x800x1280xf32>
}

// -----

// CHECK-LABEL: @InputQuantizationRestorationWithSubtract
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @InputQuantizationRestorationWithSubtract(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x800x1280xf32> {
    %zero = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %cvt = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %sub = IE.Subtract(%cvt, %zero) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %avgpool = IE.AvgPool(%sub) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    return %avgpool : tensor<1x4x800x1280xf32>

    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.540000e+02> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[AVGPOOL:%.+]] = IE.AvgPool([[FAKEQUANTIZE]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    // CHECK: return [[AVGPOOL]] : tensor<1x4x800x1280xf32>
}

// -----

// CHECK-LABEL: @ConvSub_SubHasTwoUses
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @ConvSub_SubHasTwoUses(%arg0: tensor<1x4x1600x2560xui8>) -> (tensor<1x4x800x1280xf32>, tensor<1x4x800x1280xf32>) {
    %scale1 = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %scale2 = const.Declare tensor<1x1x1x1xf32> = dense<0.00782499462> : tensor<1x1x1x1xf32>
    %zero1 = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
    %zero2 = const.Declare tensor<1x1x1x1xf32> = dense<4.0> : tensor<1x1x1x1xf32>
    %cvt = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %sub1 = IE.Subtract(%cvt, %zero1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %sub2 = IE.Subtract(%cvt, %zero2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %mul1 = IE.Multiply(%sub1, %scale1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %mul2 = IE.Multiply(%sub2, %scale2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %avgpool1 = IE.AvgPool(%mul1) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    %avgpool2 = IE.AvgPool(%mul2) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    return %avgpool1, %avgpool2 : tensor<1x4x800x1280xf32>, tensor<1x4x800x1280xf32>

    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.0312999785> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.96407366> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.00391249731> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH2:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.993774294> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FQ1:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW2]], [[OUT_HIGH2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FQ2:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW1]], [[OUT_HIGH1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[AVGPOOL1:%.+]] = IE.AvgPool([[FQ1]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    // CHECK: [[AVGPOOL2:%.+]] = IE.AvgPool([[FQ2]]) {kernel_size = [2, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 2]} : tensor<1x4x1600x2560xf32> -> tensor<1x4x800x1280xf32>
    // CHECK: return [[AVGPOOL1]], [[AVGPOOL2]] : tensor<1x4x800x1280xf32>, tensor<1x4x800x1280xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationForWAI
// CHECK-SAME: [[INPUT0:%.+]]: tensor<1x4x1600x2560xf32>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<4x4x3x3xui8>
func.func @NoInputQuantizationRestorationForWAI(%arg0: tensor<1x4x1600x2560xf32>, %arg1: tensor<4x4x3x3xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<70.5> : tensor<1x1x1x1xf32>
    %cvt = IE.Convert(%arg1) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
    %mul = IE.Multiply(%cvt, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    %conv = IE.Convolution(%arg0, %mul) {strides = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], dilations = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %conv : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<7.050000e+01> : tensor<1x1x1x1xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT1]]) {dstElemType = f32} : tensor<4x4x3x3xui8> -> tensor<4x4x3x3xf32>
    // CHECK: [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4x3x3xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x3x3xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[INPUT0]], [[MULTIPLY]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantizationRestorationForNonLayerWithPostOp
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x4x1600x2560xui8>
func.func @NoInputQuantizationRestorationForNonLayerWithPostOp(%arg0: tensor<1x4x1600x2560xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %zero = const.Declare tensor<1x1x1x1xf32> = dense<5.0> : tensor<1x1x1x1xf32>
    %cvt = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %sub = IE.Subtract(%cvt, %zero) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %mul = IE.Multiply(%sub, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %clamp = IE.Clamp(%mul) {max = 20.000000e+00 : f64, min = 1.000000e+00 : f64} : tensor<1x4x1600x2560xf32> -> tensor<1x4x1600x2560xf32>
    return %clamp : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<5.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK: [[CVT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[SUB:%.+]] = IE.Subtract([[CVT]], [[ZP]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[SUB]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MUL]]) {max = 2.000000e+01 : f64, min = 1.000000e+00 : f64} : tensor<1x4x1600x2560xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CLAMP]] : tensor<1x4x1600x2560xf32>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InputQuantRestorationConvertTranspose
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1600x2560x4xui8>
func.func @InputQuantRestorationConvertTranspose(%arg0: tensor<1x1600x2560x4xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %weights = const.Declare tensor<4x4x3x3xf32> = dense<1.0> : tensor<4x4x3x3xf32>
    %0 = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x1600x2560x4xui8> -> tensor<1x1600x2560x4xf32>
    %1 = IE.Transpose(%0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1600x2560x4xf32> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %3 = IE.Convolution(%2, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %3 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<1.000000e+00> : tensor<4x4x3x3xf32>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f32} : tensor<1x1600x2560x4xui8> -> tensor<1x1600x2560x4xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CONVERT]]) {order_value = #NWCH} : tensor<1x1600x2560x4xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FAKEQUANTIZE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @InputQuantRestorationTransposeConvert
// CHECK-SAME: [[INPUT:%.+]]: tensor<1x1600x2560x4xui8>
func.func @InputQuantRestorationTransposeConvert(%arg0: tensor<1x1600x2560x4xui8>) -> tensor<1x4x1600x2560xf32> {
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391249731> : tensor<1x1x1x1xf32>
    %weights = const.Declare tensor<4x4x3x3xf32> = dense<1.0> : tensor<4x4x3x3xf32>
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1600x2560x4xui8> -> tensor<1x4x1600x2560xui8>
    %1 = IE.Convert(%0) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    %2 = IE.Multiply(%1, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    %3 = IE.Convolution(%2, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    return %3 : tensor<1x4x1600x2560xf32>

    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.997686803> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<4x4x3x3xf32> = dense<1.000000e+00> : tensor<4x4x3x3xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[INPUT]]) {order_value = #NWCH} : tensor<1x1600x2560x4xui8> -> tensor<1x4x1600x2560xui8>
    // CHECK: [[CONVERT:%.+]] = IE.Convert([[TRANSPOSE]]) {dstElemType = f32} : tensor<1x4x1600x2560xui8> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[FAKEQUANTIZE:%.+]] = IE.FakeQuantize([[CONVERT]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x1600x2560xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FAKEQUANTIZE]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x4x1600x2560xf32>, tensor<4x4x3x3xf32> -> tensor<1x4x1600x2560xf32>
    // CHECK: return [[CONV]] : tensor<1x4x1600x2560xf32>
}

// -----

// CHECK-LABEL: @InputQuantForConvAndAddEqualOperandsShape
// CHECK-SAME: [[INPUT0:%.+]]: tensor<1x400x400x4xui8>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<1x4x400x400xf32>
func.func @InputQuantForConvAndAddEqualOperandsShape(%arg0: tensor<1x400x400x4xui8>, %arg1: tensor<1x4x400x400xf32>) -> (tensor<1x32x200x200xf32>, tensor<1x4x400x400xf32>) {
    %cvt = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x400x400x4xui8> -> tensor<1x400x400x4xf32>
    %torder = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    %transpose = IE.Transpose(%cvt, %torder) : tensor<1x400x400x4xf32>, tensor<4xsi64> -> tensor<1x4x400x400xf32>
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391986594> : tensor<1x1x1x1xf32>
    %weights = const.Declare tensor<32x4x3x3xf32> = dense<1.0> : tensor<32x4x3x3xf32>
    %mul = IE.Multiply(%transpose, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x400x400xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x400x400xf32>
    %conv = IE.Convolution(%mul, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x4x400x400xf32>, tensor<32x4x3x3xf32> -> tensor<1x32x200x200xf32>
    %add = IE.Add(%arg1, %mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x400x400xf32>, tensor<1x4x400x400xf32> -> tensor<1x4x400x400xf32>
    return %conv, %add : tensor<1x32x200x200xf32>, tensor<1x4x400x400xf32>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<32x4x3x3xf32> = dense<1.000000e+00> : tensor<32x4x3x3xf32>
    // CHECK-DAG: [[T_ORDER:%.+]] = const.Declare tensor<4xsi64> = dense<[0, 3, 1, 2]> : tensor<4xsi64>
    // CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<-0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.999565839> : tensor<1x1x1x1xf32>
    // CHECK: [[CVT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f32} : tensor<1x400x400x4xui8> -> tensor<1x400x400x4xf32>
    // CHECK: [[TRANSPOSE:%.+]] = IE.Transpose([[CVT]], [[T_ORDER]]) : tensor<1x400x400x4xf32>, tensor<4xsi64> -> tensor<1x4x400x400xf32>
    // CHECK: [[FQ:%.+]] = IE.FakeQuantize([[TRANSPOSE]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x4x400x400xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x400x400xf32>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ]], [[WEIGHTS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]} : tensor<1x4x400x400xf32>, tensor<32x4x3x3xf32> -> tensor<1x32x200x200xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[INPUT1]], [[FQ]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x400x400xf32>, tensor<1x4x400x400xf32> -> tensor<1x4x400x400xf32>
    // CHECK: return [[CONV]], [[ADD]] : tensor<1x32x200x200xf32>, tensor<1x4x400x400xf32>
}

// -----

// CHECK-LABEL: @NoInputQuantForSecondAddInputWithUnequalShape
// CHECK-SAME: [[INPUT0:%.+]]: tensor<1x4x1x1xui8>,
// CHECK-SAME: [[INPUT1:%.+]]: tensor<1x4x400x400xf32>
func.func @NoInputQuantForSecondAddInputWithUnequalShape(%arg0: tensor<1x4x1x1xui8>, %arg1: tensor<1x4x400x400xf32>) -> tensor<1x4x400x400xf32> {
    %cvt = IE.Convert(%arg0) {dstElemType = f32} : tensor<1x4x1x1xui8> -> tensor<1x4x1x1xf32>
    %scale = const.Declare tensor<1x1x1x1xf32> = dense<0.00391986594> : tensor<1x1x1x1xf32>
    %mul = IE.Multiply(%cvt, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1x1xf32>
    %add = IE.Add(%arg1, %mul) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x400x400xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x400x400xf32>
    return %add : tensor<1x4x400x400xf32>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.00391986594> : tensor<1x1x1x1xf32>
    // CHECK: [[CVT:%.+]] = IE.Convert([[INPUT0]]) {dstElemType = f32} : tensor<1x4x1x1xui8> -> tensor<1x4x1x1xf32>
    // CHECK: [[MUL:%.+]] = IE.Multiply([[CVT]], [[SCALE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x4x1x1xf32>
    // CHECK: [[ADD:%.+]] = IE.Add([[INPUT1]], [[MUL]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x400x400xf32>, tensor<1x4x1x1xf32> -> tensor<1x4x400x400xf32>
    // CHECK: return [[ADD]] : tensor<1x4x400x400xf32>
}
