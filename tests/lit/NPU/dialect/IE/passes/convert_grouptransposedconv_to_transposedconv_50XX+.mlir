//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-group-transposed-conv-to-transposed-conv="enable-sep-transposed-conv=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertGroupTransposedConvToTransposedConvQuantizedF8E4M3FN
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToTransposedConvQuantizedF8E4M3FN(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
        } : tensor<1x64x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x64x64xf16>

    %weights = const.Declare tensor<2x32x32x4x4xf16> = dense<1.000000e+00> : tensor<2x32x32x4x4xf16>
    %weights_low = const.Declare tensor<1x1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1x1xf16>
    %weights_high = const.Declare tensor<1x1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1x1xf16>
    %weights_fq = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
        } : tensor<2x32x32x4x4xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<2x32x32x4x4xf16>

    %output = IE.GroupTransposedConvolution(%input_fq, %weights_fq) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<2x32x32x4x4xf16> -> tensor<1x64x130x130xf16>

    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
        } : tensor<1x64x130x130xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x130x130xf16>

    return %output_fq : tensor<1x64x130x130xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS_SLICE0:%.+]] = const.Declare tensor<32x32x4x4xf16> = dense<1.000000e+00> : tensor<2x32x32x4x4xf16>, [#const.SubView<[0, 0, 0, 0, 0], [1, 32, 32, 4, 4]>, #const.Reshape<[32, 32, 4, 4]>]
    // CHECK-DAG:   [[WEIGHTS_SLICE1:%.+]] = const.Declare tensor<32x32x4x4xf16> = dense<1.000000e+00> : tensor<2x32x32x4x4xf16>, [#const.SubView<[1, 0, 0, 0, 0], [1, 32, 32, 4, 4]>, #const.Reshape<[32, 32, 4, 4]>]

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x64x64xf16>

    // CHECK:       [[INPUT_SLICE0:%.+]] = IE.Slice [[INPUT_FQ]] [0, 0, 0, 0] [1, 32, 64, 64] : tensor<1x64x64x64xf16> to tensor<1x32x64x64xf16>
    // CHECK:       [[WEIGHTS_SLICE0_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS_SLICE0]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<32x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x32x4x4xf16>
    // CHECK:       [[OUTPUT_SLICE0:%.+]] = IE.TransposedConvolution([[INPUT_SLICE0]], [[WEIGHTS_SLICE0_FQ]]) {
    // CHECK:               dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK:           } : tensor<1x32x64x64xf16>, tensor<32x32x4x4xf16> -> tensor<1x32x130x130xf16>

    // CHECK:       [[INPUT_SLICE1:%.+]] = IE.Slice [[INPUT_FQ]] [0, 32, 0, 0] [1, 32, 64, 64] : tensor<1x64x64x64xf16> to tensor<1x32x64x64xf16>
    // CHECK:       [[WEIGHTS_SLICE1_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS_SLICE1]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<32x32x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<32x32x4x4xf16>
    // CHECK:       [[OUTPUT_SLICE1:%.+]] = IE.TransposedConvolution([[INPUT_SLICE1]], [[WEIGHTS_SLICE1_FQ]]) {
    // CHECK-SAME:          dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:      } : tensor<1x32x64x64xf16>, tensor<32x32x4x4xf16> -> tensor<1x32x130x130xf16>

    // CHECK:       [[OUTPUT:%.+]] = IE.Concat([[OUTPUT_SLICE0]], [[OUTPUT_SLICE1]])
    // CHECK-SAME{LITERAL}:  static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]
    // CHECK-SAME:      : tensor<1x32x130x130xf16>, tensor<1x32x130x130xf16> -> tensor<1x64x130x130xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x64x130x130xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x130x130xf16>
    // CHECK:       return [[OUTPUT_FQ]]
}

// -----

// CHECK-LABEL:  func.func @ConvertGroupTransposedConvToTransposedConvDepthwiseQuantizedF8E5M2
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x64x64xf16>)
func.func @ConvertGroupTransposedConvToTransposedConvDepthwiseQuantizedF8E5M2(%input: tensor<1x64x64x64xf16>) -> tensor<1x64x130x130xf16> {
    %input_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %input_high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    %input_fq = IE.FakeQuantize(%input, %input_low, %input_high, %input_low, %input_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
        } : tensor<1x64x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x64x64xf16>

    %weights = const.Declare tensor<64x1x1x4x4xf16> = dense<1.000000e+00> : tensor<64x1x1x4x4xf16>
    %weights_low = const.Declare tensor<1x1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1x1xf16>
    %weights_high = const.Declare tensor<1x1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1x1xf16>
    %weights_fq = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
        } : tensor<64x1x1x4x4xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<64x1x1x4x4xf16>

    %output = IE.GroupTransposedConvolution(%input_fq, %weights_fq) {
            dilations = [1, 1], spatial_output_padding = [0, 0], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 2]
        } : tensor<1x64x64x64xf16>, tensor<64x1x1x4x4xf16> -> tensor<1x64x130x130xf16>

    %output_low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %output_high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    %output_fq = IE.FakeQuantize(%output, %output_low, %output_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
        } : tensor<1x64x130x130xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x130x130xf16>

    return %output_fq : tensor<1x64x130x130xf16>

    // CHECK-DAG:   [[INPUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[INPUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<64x64x4x4xf16>

    // CHECK:       [[INPUT_FQ:%.+]] = IE.FakeQuantize([[INPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x64x64x64xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x64x64xf16>
    // CHECK:       [[WEIGHTS_FQ:%.+]] = IE.FakeQuantize([[WEIGHTS]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<64x64x4x4xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<64x64x4x4xf16>

    // CHECK:       [[OUTPUT:%.+]] = IE.TransposedConvolution([[INPUT_FQ]], [[WEIGHTS_FQ]]) {
    // CHECK-SAME:      dilations = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 0>, pads_begin = [0, 0], pads_end = [0, 0], spatial_output_padding = [0, 0], strides = [2, 2]
    // CHECK-SAME:  } : tensor<1x64x64x64xf16>, tensor<64x64x4x4xf16> -> tensor<1x64x130x130xf16>

    // CHECK:       [[OUTPUT_FQ:%.+]] = IE.FakeQuantize([[OUTPUT]], [[INPUT_LOW]], [[INPUT_HIGH]], [[INPUT_LOW]], [[INPUT_HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x64x130x130xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x64x130x130xf16>

    // CHECK:       return [[OUTPUT_FQ]]
}
