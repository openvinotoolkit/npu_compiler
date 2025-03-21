//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --handle-large-pads %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

// CHECK-LABEL: @HandleLargePadsAvgPool
func.func @HandleLargePadsAvgPool(%arg0: tensor<1x16x72x128xf16>) -> tensor<1x16x75x132xf16> {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [2, 1],
        pads_begin = [1, 0],
        pads_end = [3, 4],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x72x128xf16> -> tensor<1x16x75x132xf16>

    return %ave_pool : tensor<1x16x75x132xf16>

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x16x72x4xf16> = dense<0.000000e+00> : tensor<1x16x72x4xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat(%arg0, [[CST_0]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x72x128xf16>, tensor<1x16x72x4xf16> -> tensor<1x16x72x132xf16>
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x16x2x132xf16> = dense<0.000000e+00> : tensor<1x16x2x132xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CONCAT0]], [[CST_1]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x72x132xf16>, tensor<1x16x2x132xf16> -> tensor<1x16x74x132xf16>
    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[CONCAT1]]) {kernel_size = [2, 1], pads_begin = [1, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x74x132xf16> -> tensor<1x16x75x132xf16>
    // CHECK:        return  [[AVGPOOL]]
}

// -----

// CHECK-LABEL: @HandleLargePadsMaxPool
func.func @HandleLargePadsMaxPool(%arg0: tensor<1x16x72x128xf16>) -> tensor<1x16x76x135xf16> {
    %ave_pool = IE.MaxPool(%arg0) {
        kernel_size = [2, 1],
        pads_begin = [3, 4],
        pads_end = [2, 3],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x72x128xf16> -> tensor<1x16x76x135xf16>

    return %ave_pool : tensor<1x16x76x135xf16>

    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x16x72x4xf16> = dense<0.000000e+00> : tensor<1x16x72x4xf16>
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x16x72x3xf16> = dense<0.000000e+00> : tensor<1x16x72x3xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST_0]], %arg0, [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x72x4xf16>, tensor<1x16x72x128xf16>, tensor<1x16x72x3xf16> -> tensor<1x16x72x135xf16>
    // CHECK-DAG:       [[CST_2:%.*]] = const.Declare tensor<1x16x2x135xf16> = dense<0.000000e+00> : tensor<1x16x2x135xf16>
    // CHECK-DAG:       [[CST_3:%.*]] = const.Declare tensor<1x16x1x135xf16> = dense<0.000000e+00> : tensor<1x16x1x135xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CST_2]], [[CONCAT0]], [[CST_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x2x135xf16>, tensor<1x16x72x135xf16>, tensor<1x16x1x135xf16> -> tensor<1x16x75x135xf16>
    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[CONCAT1]]) {kernel_size = [2, 1], pads_begin = [1, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x75x135xf16> -> tensor<1x16x76x135xf16>
    // CHECK:        return  [[MAXPOOL]]
}

// -----

// CHECK-LABEL: @HandleLargePadsConv
func.func @HandleLargePadsConv(%arg0: tensor<1x16x72x128xf16>) -> tensor<1x32x78x136xf16> {
    %0 = const.Declare tensor<32x16x3x3xf16> = dense<1.0> : tensor<32x16x3x3xf16>
    %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [3, 4], pads_end = [5, 6], strides = [1, 1]}
                     : tensor<1x16x72x128xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x78x136xf16>

    return %1 : tensor<1x32x78x136xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<32x16x3x3xf16> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x16x72x3xf16> = dense<0.000000e+00> : tensor<1x16x72x3xf16>
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x16x72x5xf16> = dense<0.000000e+00> : tensor<1x16x72x5xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST_0]], %arg0, [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x72x3xf16>, tensor<1x16x72x128xf16>, tensor<1x16x72x5xf16> -> tensor<1x16x72x136xf16>
    // CHECK-DAG:       [[CST_2:%.*]] = const.Declare tensor<1x16x2x136xf16> = dense<0.000000e+00> : tensor<1x16x2x136xf16>
    // CHECK-DAG:       [[CST_3:%.*]] = const.Declare tensor<1x16x4x136xf16> = dense<0.000000e+00> : tensor<1x16x4x136xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CST_2]], [[CONCAT0]], [[CST_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x2x136xf16>, tensor<1x16x72x136xf16>, tensor<1x16x4x136xf16> -> tensor<1x16x78x136xf16>
    // CHECK:       [[CONV:%.*]] = IE.Convolution([[CONCAT1]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x78x136xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x78x136xf16>
    // CHECK:        return  [[CONV]]
}

// -----

// CHECK-LABEL: @HandleLargePadsGroupConv
func.func @HandleLargePadsGroupConv(%arg0: tensor<1x16x72x128xf16>) -> tensor<1x16x78x136xf16> {
    %0 = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %1 = IE.GroupConvolution(%arg0, %0) {dilations = [1, 1], groups = 16, pads_begin = [3, 4], pads_end = [5, 6], strides = [1, 1]}
                     : tensor<1x16x72x128xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x78x136xf16>

    return %1 : tensor<1x16x78x136xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x16x72x3xf16> = dense<0.000000e+00> : tensor<1x16x72x3xf16>
    // CHECK-DAG:       [[CST_1:%.*]] = const.Declare tensor<1x16x72x5xf16> = dense<0.000000e+00> : tensor<1x16x72x5xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat([[CST_0]], %arg0, [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x16x72x3xf16>, tensor<1x16x72x128xf16>, tensor<1x16x72x5xf16> -> tensor<1x16x72x136xf16>
    // CHECK-DAG:       [[CST_2:%.*]] = const.Declare tensor<1x16x2x136xf16> = dense<0.000000e+00> : tensor<1x16x2x136xf16>
    // CHECK-DAG:       [[CST_3:%.*]] = const.Declare tensor<1x16x4x136xf16> = dense<0.000000e+00> : tensor<1x16x4x136xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CST_2]], [[CONCAT0]], [[CST_3]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x16x2x136xf16>, tensor<1x16x72x136xf16>, tensor<1x16x4x136xf16> -> tensor<1x16x78x136xf16>
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[CONCAT1]], [[CST]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x78x136xf16>, tensor<16x1x3x3xf16> -> tensor<1x16x78x136xf16>
    // CHECK:        return  [[GROUPCONV]]
}

// -----

// CHECK-LABEL: @HandleLargePadsGroupConvSmallGroup
func.func @HandleLargePadsGroupConvSmallGroup(%arg0: tensor<1x2x2x96xf16>) -> tensor<1x64x2x96xf16> {
    %0 = const.Declare tensor<64x1x3x3xf16> = dense<1.0> : tensor<64x1x3x3xf16>
    %1 = IE.GroupConvolution(%arg0, %0) {dilations = [1, 1], groups = 2 : i64, pads_begin = [2, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x2x2x96xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x2x96xf16>
    return %1 : tensor<1x64x2x96xf16>

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<64x1x3x3xf16> = dense<1.000000e+00> : tensor<64x1x3x3xf16>
    // CHECK:       [[CONCAT0:%.*]] = IE.Concat(%arg0) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x2x96xf16> -> tensor<1x2x2x96xf16>
    // CHECK-DAG:       [[CST_0:%.*]] = const.Declare tensor<1x2x1x96xf16> = dense<0.000000e+00> : tensor<1x2x1x96xf16>
    // CHECK:       [[CONCAT1:%.*]] = IE.Concat([[CST_0]], [[CONCAT0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x2x1x96xf16>, tensor<1x2x2x96xf16> -> tensor<1x2x3x96xf16>
    // CHECK:       [[GROUPCONV:%.*]] = IE.GroupConvolution([[CONCAT1]], [[CST]]) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x2x3x96xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x2x96xf16>
    // CHECK:        return  [[GROUPCONV]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolDifferentOrder
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x1x2x6xf16, {order = #NHWC}>
func.func @MaxPoolDifferentOrder(%arg0: tensor<1x1x2x6xf16, {order = #NHWC}> ) -> tensor<1x1x1x8xf16> {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 5],
        pads_begin = [0, 3],
        pads_end = [0, 3],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [2, 1]
    } : tensor<1x1x2x6xf16, {order = #NHWC}>
      -> tensor<1x1x1x8xf16>

    return %max_pool : tensor<1x1x1x8xf16>

    // CHECK-DAG:    [[CST_0:%.+]] = const.Declare tensor<1x1x2x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x2x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[CST_1:%.+]] = const.Declare tensor<1x1x2x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x1x2x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[CONCAT_0:%.+]] = IE.Concat([[CST_0]], [[ARG0]], [[CST_1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x2x1xf16, {order = #NHWC}>, tensor<1x1x2x6xf16, {order = #NHWC}>, tensor<1x1x2x1xf16, {order = #NHWC}> -> tensor<1x1x2x8xf16, {order = #NHWC}>
    // CHECK:        [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x2x8xf16, {order = #NHWC}> -> tensor<1x1x2x8xf16, {order = #NHWC}>
    // CHECK:        [[MAXPOOL:%.+]] = IE.MaxPool([[CONCAT_1]]) {kernel_size = [1, 5], pads_begin = [0, 2], pads_end = [0, 2], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1]} : tensor<1x1x2x8xf16, {order = #NHWC}> -> tensor<1x1x1x8xf16>

    // CHECK:       return [[MAXPOOL]] : tensor<1x1x1x8xf16>

}
