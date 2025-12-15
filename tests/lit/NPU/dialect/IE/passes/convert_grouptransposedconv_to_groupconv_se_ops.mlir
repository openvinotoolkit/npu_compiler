//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-group-transposed-conv-to-groupconv="enable-sep-transposed-conv=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: func.func @DoNotConvertGroupTransposedConvToGroupConv
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @DoNotConvertGroupTransposedConvToGroupConv(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>
    return %out : tensor<1x64x130x130xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// -----

// CHECK-LABEL: func.func @DoNotConvertGroupTransposedConvToGroupConvWithPadding
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @DoNotConvertGroupTransposedConvToGroupConvWithPadding(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x128x128xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [1, 1], pads_end = [1, 1], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x128x128xf16>
    return %out : tensor<1x64x128x128xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// -----

// CHECK-LABEL: func.func @DoNotConvertGroupTransposedConvToGroupConvWithOutputPadding
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @DoNotConvertGroupTransposedConvToGroupConvWithOutputPadding(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x131x131xf16> {
    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x131x131xf16>
    return %out : tensor<1x64x131x131xf16>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare
    // CHECK-NOT:  IE.Upsampling
    // CHECK-NOT:  IE.GroupConvolution
    // CHECK:      [[OUT:%.+]] = IE.GroupTransposedConvolution([[INPUT]], [[WEIGHTS]])
    // CHECK:      return [[OUT]]
}

// -----

// CHECK-LABEL: func.func @ConvertGroupTransposedConvToGroupConvLargeKernelSize
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToGroupConvLargeKernelSize(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x142x142xf16> {
    %weights = const.Declare tensor<64x1x1x16x16xf16> = dense<1.000000e+00> : tensor<64x1x1x16x16xf16>
    %out = IE.GroupTransposedConvolution(%input, %weights) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x16x16xf16> -> tensor<1x64x142x142xf16>
    return %out : tensor<1x64x142x142xf16>

    // CHECK:  [[UPSAMPLING:%.+]] = IE.Upsampling([[INPUT]])
    // CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<64x1x16x16xf16>
    // CHECK:  [[OUT:%.+]] = IE.GroupConvolution([[UPSAMPLING]], [[WEIGHTS]])
    // CHECK:  return [[OUT]]
}
