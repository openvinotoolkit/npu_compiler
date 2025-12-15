//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-avg-pool-to-dw-conv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertAveragePoolingToGroupConvolution1
func.func @ConvertAveragePoolingToGroupConvolution1(%arg0: tensor<1x2048x7x7xf16>) -> tensor<1x2048x1x1xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [7, 7],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x2048x7x7xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>

    // CHECK-NOT:   IE.AvgPool
    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<2048x1x7x7xf16> = dense<2.040100e-02> : tensor<2048x1x7x7xf16>
    // CHECK:       %[[CONV:.*]] = IE.GroupConvolution(%arg0, %[[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x7x7xf16>, tensor<2048x1x7x7xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertAveragePoolingToGroupConvolution2
func.func @ConvertAveragePoolingToGroupConvolution2() -> tensor<1x2048x1x1xf16> {
    %input = const.Declare tensor<1x2048x7x7xf16> = dense<1.000000e+00> : tensor<1x2048x7x7xf16>
    %ave_pool = IE.AvgPool(%input) {
        exclude_pads,
        kernel_size = [7, 7],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x2048x7x7xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>

    // CHECK-NOT:   IE.AvgPool
    // CHECK-DAG:       %[[INPUT:.*]] = const.Declare tensor<1x2048x7x7xf16> = dense<1.000000e+00> : tensor<1x2048x7x7xf16>
    // CHECK-DAG:       %[[WEIGHTS:.*]] = const.Declare tensor<2048x1x7x7xf16> = dense<2.040100e-02> : tensor<2048x1x7x7xf16>
    // CHECK:       %[[CONV:.*]] = IE.GroupConvolution(%[[INPUT]], %[[WEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2048
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x7x7xf16>, tensor<2048x1x7x7xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       return %[[CONV]]
}

// CHECK-LABEL: @ConvertQuantizedAveragePoolingToQuantizedGroupConvolution
func.func @ConvertQuantizedAveragePoolingToQuantizedGroupConvolution(%arg0 : tensor<1x2048x7x7xf16>) -> tensor<1x2048x1x1xf16> {
    %cst = const.Declare tensor<f16> = dense<3.662110e+00> : tensor<f16>
    %cst_0 = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    %cst_1 = const.Declare tensor<f16> = dense<6.000000e+00> : tensor<f16>
    %quantized_input = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2048x7x7xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x2048x7x7xf16>
    %ave_pool = IE.AvgPool(%quantized_input) {
        exclude_pads,
        kernel_size = [7, 7],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x2048x7x7xf16> -> tensor<1x2048x1x1xf16>
    %result = IE.FakeQuantize(%ave_pool, %cst_0, %cst, %cst_0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        levels = 256 : i64
    } : tensor<1x2048x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x2048x1x1xf16>

    return %result : tensor<1x2048x1x1xf16>

    // CHECK-NOT:   IE.AvgPool
    // CHECK-DAG:       %[[cst:.*]] = const.Declare tensor<f16> = dense<3.662110e+00> : tensor<f16>
    // CHECK-DAG:       %[[cst_0:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG:       %[[cst_1:.*]] = const.Declare tensor<f16> = dense<6.000000e+00> : tensor<f16>
    // CHECK:       %[[QINPUT:.*]] = IE.FakeQuantize(%arg0, %[[cst_0]], %[[cst_1]], %[[cst_0]], %[[cst_1]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      : tensor<1x2048x7x7xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x2048x7x7xf16>
    // CHECK-DAG:       %[[cst_2:.*]] = const.Declare tensor<2048x1x7x7xf16> = dense<1.000000e+00> : tensor<2048x1x7x7xf16>
    // CHECK-DAG:       %[[cst_3:.*]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG:       %[[cst_4:.*]] = const.Declare tensor<f16> = dense<2.550000e+02> : tensor<f16>
    // CHECK-DAG:       %[[cst_5:.*]] = const.Declare tensor<f16> = dense<5.203130e+00> : tensor<f16>
    // CHECK:       %[[QWEIGHTS:.*]] = IE.FakeQuantize(%[[cst_2]], %[[cst_3]], %[[cst_4]], %[[cst_3]], %[[cst_5]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      : tensor<2048x1x7x7xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<2048x1x7x7xf16>
    // CHECK:       %[[CONV:.*]] = IE.GroupConvolution(%[[QINPUT]], %[[QWEIGHTS]])
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      groups = 2048 : i64,
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x7x7xf16>, tensor<2048x1x7x7xf16> -> tensor<1x2048x1x1xf16>
    // CHECK:       %[[QRESULT:.*]] = IE.FakeQuantize(%[[CONV]], %[[cst_0]], %[[cst]], %[[cst_0]], %[[cst]])
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    // CHECK-SAME:      levels = 256 : i64
    // CHECK-SAME:      : tensor<1x2048x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<1x2048x1x1xf16>
    // CHECK:       return %[[QRESULT]]
}
