//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --merge-fully-connected %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @MergeMatMulWithWeightsAsConstant
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x256xf16>
func.func @MergeMatMulWithWeightsAsConstant(%arg0: tensor<1x1x256xf16>) -> tensor<1x2x1x4096xf16> {
    %weights_0 = const.Declare tensor<1x128x4096xf16> = dense<1> : tensor<4096x2x128xui4>, [#const.SubView<[0, 0, 0], [4096, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_0 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_0 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_0 = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]
    %out_high_0 = const.Declare tensor<1x1x4096xf16> = dense<2.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]

    %weights_1 = const.Declare tensor<1x128x4096xf16> = dense<1> : tensor<4096x2x128xui4>, [#const.SubView<[0, 1, 0], [4096, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_1 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_1 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_1 = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]
    %out_high_1 = const.Declare tensor<1x1x4096xf16> = dense<2.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]

    %source = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>

    %in_0 = IE.Slice %source [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %fq_0 = IE.FakeQuantize(%weights_0, %in_low_0, %in_high_0, %out_low_0, %out_high_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x128x4096xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4096xf16>, tensor<1x1x4096xf16> -> tensor<1x128x4096xf16>
    %reshape_0 = IE.Reshape(%fq_0) {shape_value = [128, 4096]} : tensor<1x128x4096xf16> -> tensor<128x4096xf16>
    %transpose_0 = IE.Transpose(%reshape_0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x4096xf16> -> tensor<4096x128xf16>
    %fc_0 = IE.FullyConnected(%in_0, %transpose_0) : tensor<1x128xf16>, tensor<4096x128xf16> -> tensor<1x4096xf16>
    %out_reshape_0 = IE.Reshape(%fc_0) {shape_value = [1, 1, 1, 4096]} : tensor<1x4096xf16> -> tensor<1x1x1x4096xf16>

    %in_1 = IE.Slice %source [0, 128] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %fq_1 = IE.FakeQuantize(%weights_1, %in_low_1, %in_high_1, %out_low_1, %out_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x128x4096xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4096xf16>, tensor<1x1x4096xf16> -> tensor<1x128x4096xf16>
    %reshape_1 = IE.Reshape(%fq_1) {shape_value = [128, 4096]} : tensor<1x128x4096xf16> -> tensor<128x4096xf16>
    %transpose_1 = IE.Transpose(%reshape_1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x4096xf16> -> tensor<4096x128xf16>
    %fc_1 = IE.FullyConnected(%in_1, %transpose_1) : tensor<1x128xf16>, tensor<4096x128xf16> -> tensor<1x4096xf16>
    %out_reshape_1 = IE.Reshape(%fc_1) {shape_value = [1, 1, 1, 4096]} : tensor<1x4096xf16> -> tensor<1x1x1x4096xf16>

    %concat = IE.Concat(%out_reshape_0, %out_reshape_1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x2x1x4096xf16>
    return %concat : tensor<1x2x1x4096xf16>

    // CHECK-DAG:      [[WEIGHTS_0:%.+]] = const.Declare tensor<1x128x4096xf16> = dense<1> : tensor<4096x2x128xui4>, [#const.SubView<[0, 0, 0], [4096, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    // CHECK-DAG:      [[WEIGHTS_1:%.+]] = const.Declare tensor<1x128x4096xf16> = dense<1> : tensor<4096x2x128xui4>, [#const.SubView<[0, 1, 0], [4096, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    // CHECK-DAG:      [[FQ_OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<2.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]
    // CHECK:      [[FQ_INPUT_LOW:%.+]] = const.Declare tensor<1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>, #const.Transpose<#map>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [1, 1]>]
    // CHECK:      [[FQ_INPUT_HIGHT:%.+]] = const.Declare tensor<1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>, #const.Transpose<#map>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [1, 1]>]
    // CHECK:      [[FQ_OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x4096xf16> = dense<-1.000000e+00> : tensor<4096x2x1xf16>, [#const.SubView<[0, 0, 0], [4096, 1, 1]>, #const.Transpose<#HWC>]

    // CHECK:      [[FQ_OUTPUT_LOW_CONCAT:%.+]] = IE.Concat([[FQ_OUTPUT_LOW]], [[FQ_OUTPUT_LOW]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x4096xf16>, tensor<1x1x4096xf16> -> tensor<2x1x4096xf16>
    // CHECK:      [[FQ_OUTPUT_LOW_TRANSPOSE:%.+]] = IE.Transpose([[FQ_OUTPUT_LOW_CONCAT]]) {order_value = #map} : tensor<2x1x4096xf16> -> tensor<2x4096x1xf16>
    // CHECK:      [[FQ_OUTPUT_LOW_RESHAPE:%.+]] = IE.AffineReshape([[FQ_OUTPUT_LOW_TRANSPOSE]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [8192, 1]} : tensor<2x4096x1xf16> -> tensor<8192x1xf16>

    // CHECK:      [[FQ_OUTPUT_HIGH_CONCAT:%.+]] = IE.Concat([[FQ_OUTPUT_HIGH]], [[FQ_OUTPUT_HIGH]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x4096xf16>, tensor<1x1x4096xf16> -> tensor<2x1x4096xf16>
    // CHECK:      [[FQ_OUTPUT_HIGH_TRANSPOSE:%.+]] = IE.Transpose([[FQ_OUTPUT_HIGH_CONCAT]]) {order_value = #map} : tensor<2x1x4096xf16> -> tensor<2x4096x1xf16>
    // CHECK:      [[FQ_OUTPUT_HIGH_RESHAPE:%.+]] = IE.AffineReshape([[FQ_OUTPUT_HIGH_TRANSPOSE]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [8192, 1]} : tensor<2x4096x1xf16> -> tensor<8192x1xf16>

    // CHECK:      [[WEIGHTS_CONCAT:%.+]] = IE.Concat([[WEIGHTS_0]], [[WEIGHTS_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x128x4096xf16>, tensor<1x128x4096xf16> -> tensor<2x128x4096xf16>
    // CHECK:      [[WEIGHTS_TRANSPOSE:%.+]] = IE.Transpose([[WEIGHTS_CONCAT]]) {order_value = #map} : tensor<2x128x4096xf16> -> tensor<2x4096x128xf16>
    // CHECK:      [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_TRANSPOSE]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [8192, 128]} : tensor<2x4096x128xf16> -> tensor<8192x128xf16>

    // CHECK:      [[INPUT_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:      [[INPUT_RESHAPE_1:%.+]] = IE.Reshape([[INPUT_RESHAPE_0]]) {shape_value = [2, 128]} : tensor<1x256xf16> -> tensor<2x128xf16>
    // CHECK:      [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS_RESHAPE]], [[FQ_INPUT_LOW]], [[FQ_INPUT_HIGHT]], [[FQ_OUTPUT_LOW_RESHAPE]], [[FQ_OUTPUT_HIGH_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<8192x128xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<8192x1xf16>, tensor<8192x1xf16> -> tensor<8192x128xf16>
    // CHECK:      [[MATMUL:%.+]] = IE.FullyConnected([[INPUT_RESHAPE_1]], [[FQ]]) : tensor<2x128xf16>, tensor<8192x128xf16> -> tensor<2x8192xf16>
    // CHECK:      [[SLICE_0:%.+]] = IE.Slice [[MATMUL]] [0, 0] [1, 4096] : tensor<2x8192xf16> to tensor<1x4096xf16>
    // CHECK:      [[OUT_RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 1, 4096]} : tensor<1x4096xf16> -> tensor<1x1x1x4096xf16>
    // CHECK:      [[SLICE_1:%.+]] = IE.Slice [[MATMUL]] [1, 4096] [1, 4096] : tensor<2x8192xf16> to tensor<1x4096xf16>
    // CHECK:      [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 1, 4096]} : tensor<1x4096xf16> -> tensor<1x1x1x4096xf16>
    // CHECK:      [[OUT_CONCAT:%.+]] = IE.Concat([[OUT_RESHAPE_0]], [[OUT_RESHAPE_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x4096xf16>, tensor<1x1x1x4096xf16> -> tensor<1x2x1x4096xf16>
    // CHECK:      return [[OUT_CONCAT]] : tensor<1x2x1x4096xf16>
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-LABEL: @DoNotMergeMatMulWithWeightsAsConstantIfZpIsNotEqual
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x1x4xf16>
func.func @DoNotMergeMatMulWithWeightsAsConstantIfZpIsNotEqual(%arg0: tensor<1x1x4xf16>) -> tensor<1x2x1x4xf16> {
    %weights_0 = const.Declare tensor<1x2x4xf16> = dense<1> : tensor<4x1x2xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_0 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_0 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_0 = const.Declare tensor<1x1x4xf16> = dense<-1.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]
    %out_high_0 = const.Declare tensor<1x1x4xf16> = dense<2.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]

    %weights_1 = const.Declare tensor<1x2x4xf16> = dense<1> : tensor<4x1x2xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_1 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_1 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_1 = const.Declare tensor<1x1x4xf16> = dense<-2.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]
    %out_high_1 = const.Declare tensor<1x1x4xf16> = dense<2.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]

    %source = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 4]} : tensor<1x1x4xf16> -> tensor<1x4xf16>

    %in_0 = IE.Slice %source [0, 0] [1, 2] : tensor<1x4xf16> to tensor<1x2xf16>
    %fq_0 = IE.FakeQuantize(%weights_0, %in_low_0, %in_high_0, %out_low_0, %out_high_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x2x4xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4xf16>, tensor<1x1x4xf16> -> tensor<1x2x4xf16>
    %reshape_0 = IE.Reshape(%fq_0) {shape_value = [2, 4]} : tensor<1x2x4xf16> -> tensor<2x4xf16>
    %transpose_0 = IE.Transpose(%reshape_0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<2x4xf16> -> tensor<4x2xf16>
    %fc_0 = IE.FullyConnected(%in_0, %transpose_0) : tensor<1x2xf16>, tensor<4x2xf16> -> tensor<1x4xf16>
    %out_reshape_0 = IE.Reshape(%fc_0) {shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>

    %in_1 = IE.Slice %source [0, 2] [1, 2] : tensor<1x4xf16> to tensor<1x2xf16>
    %fq_1 = IE.FakeQuantize(%weights_1, %in_low_1, %in_high_1, %out_low_1, %out_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x2x4xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4xf16>, tensor<1x1x4xf16> -> tensor<1x2x4xf16>
    %reshape_1 = IE.Reshape(%fq_1) {shape_value = [2, 4]} : tensor<1x2x4xf16> -> tensor<2x4xf16>
    %transpose_1 = IE.Transpose(%reshape_1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<2x4xf16> -> tensor<4x2xf16>
    %fc_1 = IE.FullyConnected(%in_1, %transpose_1) : tensor<1x2xf16>, tensor<4x2xf16> -> tensor<1x4xf16>
    %out_reshape_1 = IE.Reshape(%fc_1) {shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>

    %concat = IE.Concat(%out_reshape_0, %out_reshape_1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x4xf16>, tensor<1x1x1x4xf16> -> tensor<1x2x1x4xf16>
    return %concat : tensor<1x2x1x4xf16>

    // CHECK: [[CST:%.+]] = const.Declare tensor<1x2x4xf16> = dense<1> : tensor<4x1x2xui4>, [#const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    // CHECK: [[CST_1:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    // CHECK: [[CST_2:%.+]] = const.Declare tensor<1x1x4xf16> = dense<-1.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]
    // CHECK: [[CST_3:%.+]] = const.Declare tensor<1x1x4xf16> = dense<2.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]
    // CHECK: [[CST_4:%.+]] = const.Declare tensor<1x1x4xf16> = dense<-2.000000e+00> : tensor<4x1x1xf16>, [#const.Transpose<#HWC>]
    // CHECK: [[RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK: [[SLICE_0:%.+]] = IE.Slice [[RESHAPE_0]] [0, 0] [1, 2] : tensor<1x4xf16> to tensor<1x2xf16>
    // CHECK: [[FQ_0:%.+]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_2]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x2x4xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4xf16>, tensor<1x1x4xf16> -> tensor<1x2x4xf16>
    // CHECK: [[RESHAPE_1:%.+]] = IE.Reshape([[FQ_0]]) {shape_value = [2, 4]} : tensor<1x2x4xf16> -> tensor<2x4xf16>
    // CHECK: [[TRANSPOSE_0:%.+]] = IE.Transpose([[RESHAPE_1]]) {order_value = #CN} : tensor<2x4xf16> -> tensor<4x2xf16>
    // CHECK: [[FC_0:%.+]] = IE.FullyConnected([[SLICE_0]], [[TRANSPOSE_0]]) : tensor<1x2xf16>, tensor<4x2xf16> -> tensor<1x4xf16>
    // CHECK: [[RESHAPE_2:%.+]] = IE.Reshape([[FC_0]]) {shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>
    // CHECK: [[SLICE_1:%.+]] = IE.Slice [[RESHAPE_0]] [0, 2] [1, 2] : tensor<1x4xf16> to tensor<1x2xf16>
    // CHECK: [[FQ_1:%.+]] = IE.FakeQuantize([[CST]], [[CST_0]], [[CST_1]], [[CST_4]], [[CST_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x2x4xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x4xf16>, tensor<1x1x4xf16> -> tensor<1x2x4xf16>
    // CHECK: [[RESHAPE_3:%.+]] = IE.Reshape([[FQ_1]]) {shape_value = [2, 4]} : tensor<1x2x4xf16> -> tensor<2x4xf16>
    // CHECK: [[TRANSPOSE_1:%.+]] = IE.Transpose([[RESHAPE_3]]) {order_value = #CN} : tensor<2x4xf16> -> tensor<4x2xf16>
    // CHECK: [[FC_1:%.+]] = IE.FullyConnected([[SLICE_1]], [[TRANSPOSE_1]]) : tensor<1x2xf16>, tensor<4x2xf16> -> tensor<1x4xf16>
    // CHECK: [[RESHAPE_4:%.+]] = IE.Reshape([[FC_1]]) {shape_value = [1, 1, 1, 4]} : tensor<1x4xf16> -> tensor<1x1x1x4xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[RESHAPE_2]], [[RESHAPE_4]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x4xf16>, tensor<1x1x1x4xf16> -> tensor<1x2x1x4xf16>
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @MergeMatMulForDQPatternWithConvert
// CHECK-SAME:   [[INPUT_0:%.+]]: tensor<2x1x128xf16>, [[INPUT_1:%.+]]: tensor<2x3072x128xsi4>
func.func @MergeMatMulForDQPatternWithConvert(%arg0: tensor<2x1x128xf16>, %arg1: tensor<2x3072x128xsi4>) -> tensor<2x1x3072xf16> {
    %0:2 = IE.Split(%arg0) {axis_value = 0 : i64, num_splits = 2 : i64} : tensor<2x1x128xf16> -> tensor<1x1x128xf16>, tensor<1x1x128xf16>
    %1:2 = IE.Split(%arg1) {axis_value = 0 : i64, num_splits = 2 : i64} : tensor<2x3072x128xsi4> -> tensor<1x3072x128xsi4>, tensor<1x3072x128xsi4>

    %2 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1]], shape_value = [1, 128]} : tensor<1x1x128xf16> -> tensor<1x128xf16>
    %3 = IE.Convert(%1#1) {dstElemType = f16} : tensor<1x3072x128xsi4> -> tensor<1x3072x128xf16>
    %4 = IE.AffineReshape(%3) {dim_mapping = [[0], [0], [1]], shape_value = [3072, 128]} : tensor<1x3072x128xf16> -> tensor<3072x128xf16>
    %5 = IE.FullyConnected(%2, %4) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %6 = IE.AffineReshape(%5) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>

    %7 = IE.AffineReshape(%0#1) {dim_mapping = [[0], [0], [1]], shape_value = [1, 128]} : tensor<1x1x128xf16> -> tensor<1x128xf16>
    %8 = IE.Convert(%1#1) {dstElemType = f16} : tensor<1x3072x128xsi4> -> tensor<1x3072x128xf16>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0], [0], [1]], shape_value = [3072, 128]} : tensor<1x3072x128xf16> -> tensor<3072x128xf16>
    %10 = IE.FullyConnected(%7, %9) : tensor<1x128xf16>, tensor<3072x128xf16> -> tensor<1x3072xf16>
    %11 = IE.AffineReshape(%10) {dim_mapping = [[0, 1], [2]], shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>

    %12 = IE.Concat(%6, %11) {static_offsets = [[0, 0, 0], [1, 0, 0]]} : tensor<1x1x3072xf16>, tensor<1x1x3072xf16> -> tensor<2x1x3072xf16>
    return %12 : tensor<2x1x3072xf16>

    // CHECK:      [[SOURCE:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:                  {dim_mapping = [[0], [0], [1]], shape_value = [2, 128]} : tensor<2x1x128xf16> -> tensor<2x128xf16>
    // CHECK:      [[WEIGHTS:%.+]] = IE.Convert([[INPUT_1]]) {dstElemType = f16} : tensor<2x3072x128xsi4> -> tensor<2x3072x128xf16>
    // CHECK:      [[RESHAPE_0:%.+]] = IE.AffineReshape([[WEIGHTS]])
    // CHECK-SAME{LITERAL}:                  {dim_mapping = [[0], [0], [1]], shape_value = [6144, 128]} : tensor<2x3072x128xf16> -> tensor<6144x128xf16>
    // CHECK:      [[MATMUL:%.+]] = IE.FullyConnected([[SOURCE]], [[RESHAPE_0]]) : tensor<2x128xf16>, tensor<6144x128xf16> -> tensor<2x6144xf16>
    // CHECK:      [[SLICE_0:%.+]] = IE.Slice [[MATMUL]] [0, 0] [1, 3072] : tensor<2x6144xf16> to tensor<1x3072xf16>
    // CHECK:      [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:      [[SLICE_1:%.+]] = IE.Slice [[MATMUL]] [1, 3072] [1, 3072] : tensor<2x6144xf16> to tensor<1x3072xf16>
    // CHECK:      [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 3072]} : tensor<1x3072xf16> -> tensor<1x1x3072xf16>
    // CHECK:      [[CONCAT:%.+]] = IE.Concat([[RESHAPE_1]], [[RESHAPE_2]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x3072xf16>, tensor<1x1x3072xf16> -> tensor<2x1x3072xf16>
    // CHECK:      return [[CONCAT]] : tensor<2x1x3072xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @MergeMatMulForDQPatternWithDequantize
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x256xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<2x2560x128xsi4>
func.func @MergeMatMulForDQPatternWithDequantize(%arg0: tensor<1x1x256xf16>, %arg1: tensor<2x2560x128xsi4>) -> tensor<1x2x1x2560xf16> {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    %1 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<2x2560x128xsi4> -> tensor<2x2560x128x!qElemType>

    %2 = IE.Slice %1 [0, 0, 0] [1, 2560, 128] : tensor<2x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %3 = IE.Slice %1 [1, 0, 0] [1, 2560, 128] : tensor<2x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %4 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %5 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %6 = IE.Reshape(%4) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %7 = IE.Reshape(%5) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>

    %8 = IE.Slice %0 [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %9 = IE.Slice %0 [0, 128] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>

    %10 = IE.FullyConnected(%8, %6) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %11 = IE.FullyConnected(%9, %7) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>

    %12 = IE.Reshape(%10) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %13 = IE.Reshape(%11) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    %14 = IE.Concat(%12, %13) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x2x1x2560xf16>

    return %14 : tensor<1x2x1x2560xf16>

    // CHECK:       [[SOURCE:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOURCE]]) {shape_value = [2, 128]} : tensor<1x256xf16> -> tensor<2x128xf16>

    // CHECK:       [[WEIGHTS:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType} : tensor<2x2560x128xsi4> -> tensor<2x2560x128x!qElemType>
    // CHECK:       [[WEIGHTS_DQ:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<2x2560x128x!qElemType> -> tensor<2x2560x128xf16>
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_DQ]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [5120, 128]} : tensor<2x2560x128xf16> -> tensor<5120x128xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.FullyConnected([[RESHAPE]], [[WEIGHTS_RESHAPE]]) : tensor<2x128xf16>, tensor<5120x128xf16> -> tensor<2x5120xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[MATMUL]] [0, 0] [1, 2560] : tensor<2x5120xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[MATMUL]] [1, 2560] [1, 2560] : tensor<2x5120xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x2x1x2560xf16>
    // CHECK:       return [[CONCAT]] : tensor<1x2x1x2560xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @MergeMatMulForDQPatternWithDequantizeAndInputIsShared
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x256xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<2x2560x128xsi4>
func.func @MergeMatMulForDQPatternWithDequantizeAndInputIsShared(%arg0: tensor<1x1x256xf16>, %arg1: tensor<2x2560x128xsi4>)
-> (tensor<1x2x1x2560xf16>, tensor<1x128xf16>) {
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    %1 = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<2x2560x128xsi4> -> tensor<2x2560x128x!qElemType>

    %2 = IE.Slice %1 [0, 0, 0] [1, 2560, 128] : tensor<2x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %3 = IE.Slice %1 [1, 0, 0] [1, 2560, 128] : tensor<2x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %4 = IE.Dequantize(%2) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %5 = IE.Dequantize(%3) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %6 = IE.Reshape(%4) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %7 = IE.Reshape(%5) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>

    %8 = IE.Slice %0 [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %9 = IE.Slice %0 [0, 128] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>

    %10 = IE.FullyConnected(%8, %6) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %11 = IE.FullyConnected(%9, %7) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>

    %12 = IE.Reshape(%10) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %13 = IE.Reshape(%11) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    %14 = IE.Concat(%12, %13) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x2x1x2560xf16>

    %15 = IE.Slice %0 [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>

    return %14, %15 : tensor<1x2x1x2560xf16>, tensor<1x128xf16>

    // CHECK:       [[SOURCE:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOURCE]]) {shape_value = [2, 128]} : tensor<1x256xf16> -> tensor<2x128xf16>

    // CHECK:       [[WEIGHTS:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType} : tensor<2x2560x128xsi4> -> tensor<2x2560x128x!qElemType>
    // CHECK:       [[WEIGHTS_DQ:%.+]] = IE.Dequantize([[WEIGHTS]]) {dstElemType = f16} : tensor<2x2560x128x!qElemType> -> tensor<2x2560x128xf16>
    // CHECK:       [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_DQ]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [5120, 128]} : tensor<2x2560x128xf16> -> tensor<5120x128xf16>

    // CHECK:       [[MATMUL:%.+]] = IE.FullyConnected([[RESHAPE]], [[WEIGHTS_RESHAPE]]) : tensor<2x128xf16>, tensor<5120x128xf16> -> tensor<2x5120xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[MATMUL]] [0, 0] [1, 2560] : tensor<2x5120xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[MATMUL]] [1, 2560] [1, 2560] : tensor<2x5120xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x2x1x2560xf16>
    // CHECK:       [[OUT_SLICE:%.+]] = IE.Slice [[SOURCE]] [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    // CHECK:       return [[CONCAT]], [[OUT_SLICE]] : tensor<1x2x1x2560xf16>, tensor<1x128xf16>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @MergeMatMulForDQPatternWithDequantizeAndRegroup
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1536xf16>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<12x2560x128xsi4>
func.func @MergeMatMulForDQPatternWithDequantizeAndRegroup(%arg0: tensor<1x1x1536xf16>, %arg1: tensor<12x2560x128xsi4>) -> tensor<1x12x1x2560xf16> {
    // weights
    %weights_source = IE.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<12x2560x128xsi4> -> tensor<12x2560x128x!qElemType>
    %weights_slice_0 = IE.Slice %weights_source [0, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_1 = IE.Slice %weights_source [1, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_2 = IE.Slice %weights_source [2, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_3 = IE.Slice %weights_source [3, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_4 = IE.Slice %weights_source [4, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_5 = IE.Slice %weights_source [5, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_6 = IE.Slice %weights_source [6, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_7 = IE.Slice %weights_source [7, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_8 = IE.Slice %weights_source [8, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_9 = IE.Slice %weights_source [9, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_10 = IE.Slice %weights_source [10, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>
    %weights_slice_11 = IE.Slice %weights_source [11, 0, 0] [1, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<1x2560x128x!qElemType>

    %dq_0 = IE.Dequantize(%weights_slice_0) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_1 = IE.Dequantize(%weights_slice_1) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_2 = IE.Dequantize(%weights_slice_2) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_3 = IE.Dequantize(%weights_slice_3) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_4 = IE.Dequantize(%weights_slice_4) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_5 = IE.Dequantize(%weights_slice_5) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_6 = IE.Dequantize(%weights_slice_6) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_7 = IE.Dequantize(%weights_slice_7) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_8 = IE.Dequantize(%weights_slice_8) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_9 = IE.Dequantize(%weights_slice_9) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_10 = IE.Dequantize(%weights_slice_10) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>
    %dq_11 = IE.Dequantize(%weights_slice_11) {dstElemType = f16} : tensor<1x2560x128x!qElemType> -> tensor<1x2560x128xf16>

    %weights_reshape_0 = IE.Reshape(%dq_0) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_1 = IE.Reshape(%dq_1) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_2 = IE.Reshape(%dq_2) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_3 = IE.Reshape(%dq_3) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_4 = IE.Reshape(%dq_4) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_5 = IE.Reshape(%dq_5) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_6 = IE.Reshape(%dq_6) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_7 = IE.Reshape(%dq_7) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_8 = IE.Reshape(%dq_8) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_9 = IE.Reshape(%dq_9) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_10 = IE.Reshape(%dq_10) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>
    %weights_reshape_11 = IE.Reshape(%dq_11) {shape_value = [2560, 128]} : tensor<1x2560x128xf16> -> tensor<2560x128xf16>

    // activation
    %act_source = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 1536]} : tensor<1x1x1536xf16> -> tensor<1x1536xf16>

    %act_slice_0 = IE.Slice %act_source [0, 0] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_1 = IE.Slice %act_source [0, 128] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_2 = IE.Slice %act_source [0, 256] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_3 = IE.Slice %act_source [0, 384] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_4 = IE.Slice %act_source [0, 512] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_5 = IE.Slice %act_source [0, 640] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_6 = IE.Slice %act_source [0, 768] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_7 = IE.Slice %act_source [0, 896] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_8 = IE.Slice %act_source [0, 1024] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_9 = IE.Slice %act_source [0, 1152] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_10 = IE.Slice %act_source [0, 1280] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>
    %act_slice_11 = IE.Slice %act_source [0, 1408] [1, 128] : tensor<1x1536xf16> to tensor<1x128xf16>

    // FC layers
    %fc_0 = IE.FullyConnected(%act_slice_0, %weights_reshape_0) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_1 = IE.FullyConnected(%act_slice_1, %weights_reshape_1) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_2 = IE.FullyConnected(%act_slice_2, %weights_reshape_2) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_3 = IE.FullyConnected(%act_slice_3, %weights_reshape_3) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_4 = IE.FullyConnected(%act_slice_4, %weights_reshape_4) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_5 = IE.FullyConnected(%act_slice_5, %weights_reshape_5) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_6 = IE.FullyConnected(%act_slice_6, %weights_reshape_6) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_7 = IE.FullyConnected(%act_slice_7, %weights_reshape_7) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_8 = IE.FullyConnected(%act_slice_8, %weights_reshape_8) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_9 = IE.FullyConnected(%act_slice_9, %weights_reshape_9) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_10 = IE.FullyConnected(%act_slice_10, %weights_reshape_10) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>
    %fc_11 = IE.FullyConnected(%act_slice_11, %weights_reshape_11) : tensor<1x128xf16>, tensor<2560x128xf16> -> tensor<1x2560xf16>

    %reshape_0 = IE.Reshape(%fc_0) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_1 = IE.Reshape(%fc_1) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_2 = IE.Reshape(%fc_2) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_3 = IE.Reshape(%fc_3) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_4 = IE.Reshape(%fc_4) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_5 = IE.Reshape(%fc_5) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_6 = IE.Reshape(%fc_6) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_7 = IE.Reshape(%fc_7) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_8 = IE.Reshape(%fc_8) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_9 = IE.Reshape(%fc_9) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_10 = IE.Reshape(%fc_10) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>
    %reshape_11 = IE.Reshape(%fc_11) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    %concat = IE.Concat(%reshape_0, %reshape_1, %reshape_2, %reshape_3, %reshape_4, %reshape_5, %reshape_6, %reshape_7, %reshape_8, %reshape_9, %reshape_10, %reshape_11) {per_axis = #IE.Concat<axis = 1 : i64>} :
            tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>,
            tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x12x1x2560xf16>

    return %concat : tensor<1x12x1x2560xf16>

    // CHECK:       [[WEIGHTS_SOURCE:%.+]] = IE.QuantizeCast([[INPUT_1]]) {dstElemType = !qElemType} : tensor<12x2560x128xsi4> -> tensor<12x2560x128x!qElemType>
    // CHECK:       [[WEIGHTS_SLICE_1:%.+]] = IE.Slice [[WEIGHTS_SOURCE]] [8, 0, 0] [4, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<4x2560x128x!qElemType>
    // CHECK:       [[WEIGHTS_DQ_1:%.+]] = IE.Dequantize([[WEIGHTS_SLICE_1]]) {dstElemType = f16} : tensor<4x2560x128x!qElemType> -> tensor<4x2560x128xf16>
    // CHECK:       [[WEIGHTS_RESHAPE_1:%.+]] = IE.AffineReshape([[WEIGHTS_DQ_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [10240, 128]} : tensor<4x2560x128xf16> -> tensor<10240x128xf16>
    // CHECK:       [[WEIGHTS_SLICE_0:%.+]] = IE.Slice [[WEIGHTS_SOURCE]] [0, 0, 0] [8, 2560, 128] : tensor<12x2560x128x!qElemType> to tensor<8x2560x128x!qElemType>
    // CHECK:       [[WEIGHTS_DQ_0:%.+]] = IE.Dequantize([[WEIGHTS_SLICE_0]]) {dstElemType = f16} : tensor<8x2560x128x!qElemType> -> tensor<8x2560x128xf16>
    // CHECK:       [[WEIGHTS_RESHAPE_0:%.+]] = IE.AffineReshape([[WEIGHTS_DQ_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [20480, 128]} : tensor<8x2560x128xf16> -> tensor<20480x128xf16>

    // CHECK:       [[ACT_SOURCE:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [1]], shape_value = [1, 1536]} : tensor<1x1x1536xf16> -> tensor<1x1536xf16>

    // merged fc 0
    // CHECK:       [[ACT_SLICE_0:%.+]] = IE.Slice [[ACT_SOURCE]] [0, 0] [1, 1024] : tensor<1x1536xf16> to tensor<1x1024xf16>
    // CHECK:       [[ACT_RESHAPE_0:%.+]] = IE.Reshape([[ACT_SLICE_0]]) {shape_value = [8, 128]} : tensor<1x1024xf16> -> tensor<8x128xf16>

    // CHECK:       [[FC_0:%.+]] = IE.FullyConnected([[ACT_RESHAPE_0]], [[WEIGHTS_RESHAPE_0]]) : tensor<8x128xf16>, tensor<20480x128xf16> -> tensor<8x20480xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[FC_0]] [0, 0] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[FC_0]] [1, 2560] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[FC_0]] [2, 5120] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_2:%.+]] = IE.Reshape([[SLICE_2]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_3:%.+]] = IE.Slice [[FC_0]] [3, 7680] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_3:%.+]] = IE.Reshape([[SLICE_3]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_4:%.+]] = IE.Slice [[FC_0]] [4, 10240] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_4:%.+]] = IE.Reshape([[SLICE_4]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_5:%.+]] = IE.Slice [[FC_0]] [5, 12800] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_5:%.+]] = IE.Reshape([[SLICE_5]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_6:%.+]] = IE.Slice [[FC_0]] [6, 15360] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_6:%.+]] = IE.Reshape([[SLICE_6]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_7:%.+]] = IE.Slice [[FC_0]] [7, 17920] [1, 2560] : tensor<8x20480xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_7:%.+]] = IE.Reshape([[SLICE_7]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // merged fc 1
    // CHECK:       [[ACT_SLICE_1:%.+]] = IE.Slice [[ACT_SOURCE]] [0, 1024] [1, 512] : tensor<1x1536xf16> to tensor<1x512xf16>
    // CHECK:       [[ACT_RESHAPE_1:%.+]] = IE.Reshape([[ACT_SLICE_1]]) {shape_value = [4, 128]} : tensor<1x512xf16> -> tensor<4x128xf16>

    // CHECK:       [[FC_1:%.+]] = IE.FullyConnected([[ACT_RESHAPE_1]], [[WEIGHTS_RESHAPE_1]]) : tensor<4x128xf16>, tensor<10240x128xf16> -> tensor<4x10240xf16>

    // CHECK:       [[SLICE_8:%.+]] = IE.Slice [[FC_1]] [0, 0] [1, 2560] : tensor<4x10240xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_8:%.+]] = IE.Reshape([[SLICE_8]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_9:%.+]] = IE.Slice [[FC_1]] [1, 2560] [1, 2560] : tensor<4x10240xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_9:%.+]] = IE.Reshape([[SLICE_9]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_10:%.+]] = IE.Slice [[FC_1]] [2, 5120] [1, 2560] : tensor<4x10240xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_10:%.+]] = IE.Reshape([[SLICE_10]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[SLICE_11:%.+]] = IE.Slice [[FC_1]] [3, 7680] [1, 2560] : tensor<4x10240xf16> to tensor<1x2560xf16>
    // CHECK:       [[RESHAPE_11:%.+]] = IE.Reshape([[SLICE_11]]) {shape_value = [1, 1, 1, 2560]} : tensor<1x2560xf16> -> tensor<1x1x1x2560xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[RESHAPE_0]], [[RESHAPE_1]], [[RESHAPE_2]], [[RESHAPE_3]], [[RESHAPE_4]], [[RESHAPE_5]], [[RESHAPE_6]], [[RESHAPE_7]],
    // CHECK-SAME:                             [[RESHAPE_8]], [[RESHAPE_9]], [[RESHAPE_10]], [[RESHAPE_11]]) {per_axis = #IE.Concat<axis = 1 : i64>} :
    // CHECK-SAME:          tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>,
    // CHECK-SAME:          tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16>, tensor<1x1x1x2560xf16> -> tensor<1x12x1x2560xf16>

    // CHECK:       return [[CONCAT]] : tensor<1x12x1x2560xf16>
}
