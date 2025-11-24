//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-batch="skip-unroll-batch=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @SkipUnrollAveragePoolingBatch
// CHECK-SAME:      [[IN1:%.+]]: tensor<2x128x32x64xf16>
func.func @SkipUnrollAveragePoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %AVG_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[AVG_POOL:%.+]] = IE.AvgPool([[IN1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[AVG_POOL]] : tensor<2x128x32x64xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollMaxPoolingBatch
// CHECK-SAME:      [[IN1:%.+]]: tensor<2x128x32x64xf16>
func.func @SkipUnrollMaxPoolingBatch(%arg0: tensor<2x128x32x64xf16>) -> tensor<2x128x32x64xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    return %MAX_POOL : tensor<2x128x32x64xf16>

    // CHECK:   [[MAX_POOL:%.+]] = IE.MaxPool([[IN1]]) {
    // CHECK-SAME:      kernel_size = [3, 3],
    // CHECK-SAME:      pads_begin = [1, 1],
    // CHECK-SAME:      pads_end = [1, 1],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<2x128x32x64xf16> -> tensor<2x128x32x64xf16>

    // CHECK:   return [[MAX_POOL]] : tensor<2x128x32x64xf16>
}

// -----

// CHECK-LABEL: @SkipUnrollConvolutionBatch
// CHECK-SAME:      [[IN1:%.+]]: tensor<3x3x62x62xf32>
func.func @SkipUnrollConvolutionBatch(%arg0: tensor<3x3x62x62xf32>) -> tensor<3x48x60x60xf32> {
    %CST = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
    %CONV = IE.Convolution(%arg0, %CST) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<3x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<3x48x60x60xf32>

    return %CONV : tensor<3x48x60x60xf32>

    // CHECK:   [[CST_0:%.+]] = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> :
    // CHECK-SAME:      tensor<48x3x3x3xf32>

    // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[IN1]], [[CST_0]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<3x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<3x48x60x60xf32>

    // CHECK:   return [[CONV_0]] : tensor<3x48x60x60xf32>
}
