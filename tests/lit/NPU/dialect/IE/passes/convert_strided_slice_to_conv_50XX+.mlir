//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --mlir-print-elementsattrs-with-hex-if-larger=-1 --init-compiler="vpu-arch=%arch%" --convert-strided-slice-to-conv %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertStridedSliceWithFQ2ConvF8E4M3FN
// CHECK-SAME:      [[ARG_0:%.+]]: tensor<1x3x640x640xf16, {order = #NHWC}>
func.func @ConvertStridedSliceWithFQ2ConvF8E4M3FN(%arg0: tensor<1x3x640x640xf16, {order = #NHWC}>) -> tensor<1x3x320x640xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst, %cst_0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 640, 640], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}> -> tensor<1x3x320x640xf16, {order = #NHWC}>
    return %1 : tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK-NOT: IE.StridedSlice

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<3x3x1x1xf16>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[1.000000e+00]], [[0.000000e+00]], [[0.000000e+00]]], [[[0.000000e+00]], [[1.000000e+00]], [[0.000000e+00]]], [[[0.000000e+00]], [[0.000000e+00]], [[1.000000e+00]]]]> : tensor<3x3x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<f16> = dense<2.540000e+02> : tensor<f16>
    // CHECK-DAG: [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG: [[CST3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[ARG_0]], [[CST3]], [[CST2]], [[CST3]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x640x640xf16, {order = #NHWC}>
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[CST]], [[CST0]], [[CST1]], [[CST0]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<3x3x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<3x3x1x1xf16>
    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x3x640x640xf16, {order = #NHWC}>, tensor<3x3x1x1xf16> -> tensor<1x3x320x640xf16, {order = #NHWC}>

    // CHECK: return [[CONV]] : tensor<1x3x320x640xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ConvertParallelStridedSlicesToConvWithFakeQuantizeF8E5M2
// CHECK-SAME:      [[ARG_0:%.+]]: tensor<1x3x416x416xf16>
func.func @ConvertParallelStridedSlicesToConvWithFakeQuantizeF8E5M2(%arg0: tensor<1x3x416x416xf16>) -> tensor<1x12x208x208xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %0 = IE.FakeQuantize(%arg0, %cst_1, %cst_0, %cst_1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    %1 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %2 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %3 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %4 = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 3, 416, 416], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 2]} : tensor<1x3x416x416xf16> -> tensor<1x3x208x208xf16>
    %5 = IE.Concat(%1, %2, %3, %4) {static_offsets = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0]]} : tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16>, tensor<1x3x208x208xf16> -> tensor<1x12x208x208xf16>
    return %5 : tensor<1x12x208x208xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<12x3x2x2xf16>
    // CHECK-DAG-SAME{LITERAL}: = dense<[[[[1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[1.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [1.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 1.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]]], [[[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]]]>
    // CHECK-DAG-SAME:              tensor<12x3x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG: [[CST_0:%.+]] = const.Declare tensor<f16> = dense<0.000000e+00> : tensor<f16>
    // CHECK-DAG: [[CST_1:%.+]] = const.Declare tensor<f16> = dense<2.540000e+02> : tensor<f16>
    // CHECK-DAG: [[CST_2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK-DAG: [[CST_3:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[ARG_0]], [[CST_3]], [[CST_2]], [[CST_3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x3x416x416xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x416x416xf16>
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_0]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<12x3x2x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<12x3x2x2xf16>

    // CHECK: [[CONV:%.+]] = IE.Convolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x416x416xf16>, tensor<12x3x2x2xf16> -> tensor<1x12x208x208xf16>

    // CHECK: return [[CONV]] : tensor<1x12x208x208xf16>
}
