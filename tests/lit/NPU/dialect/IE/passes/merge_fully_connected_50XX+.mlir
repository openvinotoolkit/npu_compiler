//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --merge-fully-connected="merge-unrolled-matmul-for-large-oc=true" %s | FileCheck %s --check-prefix=CHECK-ENABLED
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --merge-fully-connected="merge-unrolled-matmul-for-large-oc=false" %s | FileCheck %s --check-prefix=CHECK-DEFAULT
// REQUIRES: platform-NPU5010

// This test exercises `merge-fully-connected` with `merge-unrolled-matmul-for-large-oc` explicitly enabled and disabled.
// Each branch has OC = 8320 (> VPU_DIMENSION_LIMIT = 8192).
// CHECK-ENABLED (option enabled): the branches are merged into a single FullyConnected and the resulting IR stays legal.
// CHECK-DEFAULT (option disabled): the pass runs with the option explicitly disabled and the branches remain unmerged.

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK-ENABLED-LABEL: @MergeMatMulLargeOCWithWeightsAsConstant
// CHECK-ENABLED-SAME:   [[INPUT:%.+]]: tensor<1x1x256xf16>
// CHECK-DEFAULT-LABEL: @MergeMatMulLargeOCWithWeightsAsConstant
func.func @MergeMatMulLargeOCWithWeightsAsConstant(%arg0: tensor<1x1x256xf16>) -> tensor<1x2x1x8320xf16> {
    %weights_0 = const.Declare tensor<1x128x8320xf16> = dense<1> : tensor<8320x2x128xui4>, [#const.SubView<[0, 0, 0], [8320, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_0 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_0 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_0 = const.Declare tensor<1x1x8320xf16> = dense<-1.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]
    %out_high_0 = const.Declare tensor<1x1x8320xf16> = dense<2.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]

    %weights_1 = const.Declare tensor<1x128x8320xf16> = dense<1> : tensor<8320x2x128xui4>, [#const.SubView<[0, 1, 0], [8320, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    %in_low_1 = const.Declare tensor<1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %in_high_1 = const.Declare tensor<1x1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>]
    %out_low_1 = const.Declare tensor<1x1x8320xf16> = dense<-1.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]
    %out_high_1 = const.Declare tensor<1x1x8320xf16> = dense<2.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]

    %source = IE.AffineReshape(%arg0) {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>

    %in_0 = IE.Slice %source [0, 0] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %fq_0 = IE.FakeQuantize(%weights_0, %in_low_0, %in_high_0, %out_low_0, %out_high_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x128x8320xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x8320xf16>, tensor<1x1x8320xf16> -> tensor<1x128x8320xf16>
    %reshape_0 = IE.Reshape(%fq_0) {shape_value = [128, 8320]} : tensor<1x128x8320xf16> -> tensor<128x8320xf16>
    %transpose_0 = IE.Transpose(%reshape_0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x8320xf16> -> tensor<8320x128xf16>
    %fc_0 = IE.FullyConnected(%in_0, %transpose_0) : tensor<1x128xf16>, tensor<8320x128xf16> -> tensor<1x8320xf16>
    %out_reshape_0 = IE.Reshape(%fc_0) {shape_value = [1, 1, 1, 8320]} : tensor<1x8320xf16> -> tensor<1x1x1x8320xf16>

    %in_1 = IE.Slice %source [0, 128] [1, 128] : tensor<1x256xf16> to tensor<1x128xf16>
    %fq_1 = IE.FakeQuantize(%weights_1, %in_low_1, %in_high_1, %out_low_1, %out_high_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<1x128x8320xf16>, tensor<1x1x1xf16>, tensor<1x1x1xf16>, tensor<1x1x8320xf16>, tensor<1x1x8320xf16> -> tensor<1x128x8320xf16>
    %reshape_1 = IE.Reshape(%fq_1) {shape_value = [128, 8320]} : tensor<1x128x8320xf16> -> tensor<128x8320xf16>
    %transpose_1 = IE.Transpose(%reshape_1) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<128x8320xf16> -> tensor<8320x128xf16>
    %fc_1 = IE.FullyConnected(%in_1, %transpose_1) : tensor<1x128xf16>, tensor<8320x128xf16> -> tensor<1x8320xf16>
    %out_reshape_1 = IE.Reshape(%fc_1) {shape_value = [1, 1, 1, 8320]} : tensor<1x8320xf16> -> tensor<1x1x1x8320xf16>

    %concat = IE.Concat(%out_reshape_0, %out_reshape_1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x8320xf16>, tensor<1x1x1x8320xf16> -> tensor<1x2x1x8320xf16>
    return %concat : tensor<1x2x1x8320xf16>

    // Option enabled: the two large-OC branches are merged into a single FullyConnected.
    // CHECK-ENABLED-DAG:      [[WEIGHTS_0:%.+]] = const.Declare tensor<1x128x8320xf16> = dense<1> : tensor<8320x2x128xui4>, [#const.SubView<[0, 0, 0], [8320, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    // CHECK-ENABLED-DAG:      [[WEIGHTS_1:%.+]] = const.Declare tensor<1x128x8320xf16> = dense<1> : tensor<8320x2x128xui4>, [#const.SubView<[0, 1, 0], [8320, 1, 128]>, #const.ConvertElemType<ui8>, #const.CastElemType<ui4>, #const.CastElemType<f16>, #const.Transpose<#HWC>]
    // CHECK-ENABLED-DAG:      [[FQ_OUTPUT_HIGH:%.+]] = const.Declare tensor<1x1x8320xf16> = dense<2.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]
    // CHECK-ENABLED:      [[FQ_INPUT_LOW:%.+]] = const.Declare tensor<1x1xf16> = dense<0.000000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>, #const.Transpose<#map>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [1, 1]>]
    // CHECK-ENABLED:      [[FQ_INPUT_HIGH:%.+]] = const.Declare tensor<1x1xf16> = dense<1.500000e+00> : tensor<1x1x1xf16>, [#const.Transpose<#HWC>, #const.Transpose<#map>, #const.AffineReshape<{{\[\[}}0], [0], [1]], [1, 1]>]
    // CHECK-ENABLED:      [[FQ_OUTPUT_LOW:%.+]] = const.Declare tensor<1x1x8320xf16> = dense<-1.000000e+00> : tensor<8320x2x1xf16>, [#const.SubView<[0, 0, 0], [8320, 1, 1]>, #const.Transpose<#HWC>]

    // CHECK-ENABLED:      [[FQ_OUTPUT_LOW_CONCAT:%.+]] = IE.Concat([[FQ_OUTPUT_LOW]], [[FQ_OUTPUT_LOW]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x8320xf16>, tensor<1x1x8320xf16> -> tensor<2x1x8320xf16>
    // CHECK-ENABLED:      [[FQ_OUTPUT_LOW_TRANSPOSE:%.+]] = IE.Transpose([[FQ_OUTPUT_LOW_CONCAT]]) {order_value = #map} : tensor<2x1x8320xf16> -> tensor<2x8320x1xf16>
    // CHECK-ENABLED:      [[FQ_OUTPUT_LOW_RESHAPE:%.+]] = IE.AffineReshape([[FQ_OUTPUT_LOW_TRANSPOSE]])
    // CHECK-ENABLED-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [16640, 1]} : tensor<2x8320x1xf16> -> tensor<16640x1xf16>

    // CHECK-ENABLED:      [[FQ_OUTPUT_HIGH_CONCAT:%.+]] = IE.Concat([[FQ_OUTPUT_HIGH]], [[FQ_OUTPUT_HIGH]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1x8320xf16>, tensor<1x1x8320xf16> -> tensor<2x1x8320xf16>
    // CHECK-ENABLED:      [[FQ_OUTPUT_HIGH_TRANSPOSE:%.+]] = IE.Transpose([[FQ_OUTPUT_HIGH_CONCAT]]) {order_value = #map} : tensor<2x1x8320xf16> -> tensor<2x8320x1xf16>
    // CHECK-ENABLED:      [[FQ_OUTPUT_HIGH_RESHAPE:%.+]] = IE.AffineReshape([[FQ_OUTPUT_HIGH_TRANSPOSE]])
    // CHECK-ENABLED-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [16640, 1]} : tensor<2x8320x1xf16> -> tensor<16640x1xf16>

    // CHECK-ENABLED:      [[WEIGHTS_CONCAT:%.+]] = IE.Concat([[WEIGHTS_0]], [[WEIGHTS_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x128x8320xf16>, tensor<1x128x8320xf16> -> tensor<2x128x8320xf16>
    // CHECK-ENABLED:      [[WEIGHTS_TRANSPOSE:%.+]] = IE.Transpose([[WEIGHTS_CONCAT]]) {order_value = #map} : tensor<2x128x8320xf16> -> tensor<2x8320x128xf16>
    // CHECK-ENABLED:      [[WEIGHTS_RESHAPE:%.+]] = IE.AffineReshape([[WEIGHTS_TRANSPOSE]])
    // CHECK-ENABLED-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [16640, 128]} : tensor<2x8320x128xf16> -> tensor<16640x128xf16>

    // CHECK-ENABLED:      [[INPUT_RESHAPE_0:%.+]] = IE.AffineReshape([[INPUT]])
    // CHECK-ENABLED-SAME{LITERAL}:          {dim_mapping = [[0], [0], [1]], shape_value = [1, 256]} : tensor<1x1x256xf16> -> tensor<1x256xf16>
    // CHECK-ENABLED:      [[INPUT_RESHAPE_1:%.+]] = IE.Reshape([[INPUT_RESHAPE_0]]) {shape_value = [2, 128]} : tensor<1x256xf16> -> tensor<2x128xf16>
    // CHECK-ENABLED:      [[FQ:%.+]] = IE.FakeQuantize([[WEIGHTS_RESHAPE]], [[FQ_INPUT_LOW]], [[FQ_INPUT_HIGH]], [[FQ_OUTPUT_LOW_RESHAPE]], [[FQ_OUTPUT_HIGH_RESHAPE]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64} : tensor<16640x128xf16>, tensor<1x1xf16>, tensor<1x1xf16>, tensor<16640x1xf16>, tensor<16640x1xf16> -> tensor<16640x128xf16>
    // CHECK-ENABLED:      [[MATMUL:%.+]] = IE.FullyConnected([[INPUT_RESHAPE_1]], [[FQ]]) : tensor<2x128xf16>, tensor<16640x128xf16> -> tensor<2x16640xf16>
    // CHECK-ENABLED:      [[SLICE_0:%.+]] = IE.Slice [[MATMUL]] [0, 0] [1, 8320] : tensor<2x16640xf16> to tensor<1x8320xf16>
    // CHECK-ENABLED:      [[OUT_RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [1, 1, 1, 8320]} : tensor<1x8320xf16> -> tensor<1x1x1x8320xf16>
    // CHECK-ENABLED:      [[SLICE_1:%.+]] = IE.Slice [[MATMUL]] [1, 8320] [1, 8320] : tensor<2x16640xf16> to tensor<1x8320xf16>
    // CHECK-ENABLED:      [[OUT_RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [1, 1, 1, 8320]} : tensor<1x8320xf16> -> tensor<1x1x1x8320xf16>
    // CHECK-ENABLED:      [[OUT_CONCAT:%.+]] = IE.Concat([[OUT_RESHAPE_0]], [[OUT_RESHAPE_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x8320xf16>, tensor<1x1x1x8320xf16> -> tensor<1x2x1x8320xf16>
    // CHECK-ENABLED:      return [[OUT_CONCAT]] : tensor<1x2x1x8320xf16>

    // Option disabled: the two large-OC branches stay unmerged (no fused 2x16640 FullyConnected).
    // CHECK-DEFAULT-NOT:   tensor<2x16640xf16>
    // CHECK-DEFAULT:       IE.FullyConnected(%{{.+}}, %{{.+}}) : tensor<1x128xf16>, tensor<8320x128xf16> -> tensor<1x8320xf16>
    // CHECK-DEFAULT:       IE.FullyConnected(%{{.+}}, %{{.+}}) : tensor<1x128xf16>, tensor<8320x128xf16> -> tensor<1x8320xf16>
    // CHECK-DEFAULT:       IE.Concat(%{{.+}}, %{{.+}}) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1x8320xf16>, tensor<1x1x1x8320xf16> -> tensor<1x2x1x8320xf16>
}
