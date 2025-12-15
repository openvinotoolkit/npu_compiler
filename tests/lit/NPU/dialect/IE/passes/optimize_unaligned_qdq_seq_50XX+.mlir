//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-unaligned-qdq-seq %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @OptimizeQuantDequantSequenceF8E4M3FN
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x40x1x1xf16>
// CHECK-SAME:    [[INPUT_1:%.+]]: tensor<512x40x1x1xf16>
func.func @OptimizeQuantDequantSequenceF8E4M3FN(%input0 : tensor<1x40x1x1xf16>, %input1 : tensor<512x40x1x1xf16>) -> tensor<1x64x1x8xf16> {
    %low = const.Declare tensor<f16> = dense<-4.480000e+02> : tensor<f16>
    %high = const.Declare tensor<f16> = dense<4.480000e+02> : tensor<f16>

    %1 = IE.Convolution(%input0, %input1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
    %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>

    %3 = IE.FakeQuantize(%2, %low, %high, %low, %high) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
    } : tensor<1x1x1x512xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x1x1x512xf16>

    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 64]} : tensor<1x1x1x512xf16> -> tensor<1x1x8x64xf16>
    %5 = IE.Transpose(%4) {order_value = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>} : tensor<1x1x8x64xf16> -> tensor<1x64x1x8xf16>

    return %5 : tensor<1x64x1x8xf16>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<f16> = dense<-4.480000e+02> : tensor<f16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<f16> = dense<4.480000e+02> : tensor<f16>

    // CHECK:    [[CONV:%.+]] = IE.Convolution([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x40x1x1xf16>, tensor<512x40x1x1xf16> -> tensor<1x512x1x1xf16>
    // CHECK:    [[FQ:%.+]] = IE.FakeQuantize([[CONV]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x512x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x512x1x1xf16>

    // CHECK:    [[RESHAPE_0:%.+]] = IE.AffineReshape([[FQ]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512x1x1xf16> -> tensor<1x1x1x512xf16>
    // CHECK:    [[RESHAPE_1:%.+]] = IE.AffineReshape([[RESHAPE_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 1, 8, 64]} : tensor<1x1x1x512xf16> -> tensor<1x1x8x64xf16>
    // CHECK:    [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE_1]]) {order_value = #NWCH} : tensor<1x1x8x64xf16> -> tensor<1x64x1x8xf16>

    // CHECK:    return [[TRANSPOSE]] : tensor<1x64x1x8xf16>
}
