//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertBilinearWithFQToStridedConcatAndConvF8E4M3FN
func.func @ConvertBilinearWithFQToStridedConcatAndConvF8E4M3FN(%arg0: tensor<1x16x96x176xf16>) -> tensor<1x16x192x352xf16> {
    %input_low = const.Declare tensor<f32> = dense<-4.480000e+02> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<4.480000e+02> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } :
        tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>

    %1 = IE.Interpolate(%0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x16x96x176xf16> -> tensor<1x16x192x352xf16>


    %2 = IE.FakeQuantize(%1, %input_low, %input_high, %input_low, %input_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN } :
        tensor<1x16x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x192x352xf16>

    return %2 : tensor<1x16x192x352xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK:       [[FQ_0:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[FQ_0]] [0, 0, 0, 175] [1, 16, 96, 1] : tensor<1x16x96x176xf16> to tensor<1x16x96x1xf16>
    // CHECK:       [[CONCAT_0:%.+]] = IE.Concat([[FQ_0]], [[SLICE_0]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 176]]} : tensor<1x16x96x176xf16>, tensor<1x16x96x1xf16> -> tensor<1x16x96x177xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[CONCAT_0]] [0, 0, 95, 0] [1, 16, 1, 177] : tensor<1x16x96x177xf16> to tensor<1x16x1x177xf16>
    // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[CONCAT_0]], [[SLICE_1]])
    // CHECK{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]} : tensor<1x16x96x177xf16>, tensor<1x16x1x177xf16> -> tensor<1x16x97x177xf16>
    // CHECK:       [[SLICE_2:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 0, 0] [1, 16, 97, 176] : tensor<1x16x97x177xf16> to tensor<1x16x97x176xf16>
    // CHECK:       [[FQ_1:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>

    // CHECK:       [[GROUPCONV0:%.+]] = IE.GroupConvolution([[FQ_0]], [[FQ_1]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_2:%.+]] = IE.FakeQuantize([[GROUPCONV0]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_3:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}} {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x1x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x2xf16>

    // CHECK:       [[GROUPCONV1:%.+]] = IE.GroupConvolution([[CONCAT_0]], [[FQ_3]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x177xf16>, tensor<16x1x1x2xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_4:%.+]] = IE.FakeQuantize([[GROUPCONV1]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_5:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x2x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x2x1xf16>

    // CHECK:       [[GROUPCONV2:%.+]] = IE.GroupConvolution([[SLICE_2]], [[FQ_5]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x97x176xf16>, tensor<16x1x2x1xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_6:%.+]] = IE.FakeQuantize([[GROUPCONV2]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_7:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x2x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x2x2xf16>

    // CHECK:       [[GROUPCONV3:%.+]] = IE.GroupConvolution([[CONCAT_1]], [[FQ_7]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x97x177xf16>, tensor<16x1x2x2xf16> -> tensor<1x16x96x176xf16>
    // CHECK:       [[FQ_8:%.+]] = IE.FakeQuantize([[GROUPCONV3]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x176xf16>

    // CHECK:       [[CONCAT_2:%.+]] = IE.Concat([[FQ_2]], [[FQ_4]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x176xf16>, tensor<1x16x96x176xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[CONCAT_3:%.+]] = IE.Concat([[FQ_6:%.+]], [[FQ_8]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x176xf16>, tensor<1x16x96x176xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ_9:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUPCONV4:%.+]] = IE.GroupConvolution([[CONCAT_2]], [[FQ_9]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ_10:%.+]] = IE.FakeQuantize([[GROUPCONV4]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x352xf16>
    // CHECK:       [[FQ_11:%.+]] = IE.FakeQuantize({{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<16x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<16x1x1x1xf16>
    // CHECK:       [[GROUPCONV5:%.+]] = IE.GroupConvolution([[CONCAT_3]], [[FQ_11]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<16x1x1x1xf16> -> tensor<1x16x96x352xf16>

    // CHECK:       [[FQ_12:%.+]] = IE.FakeQuantize([[GROUPCONV5]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x96x352xf16>
    // CHECK:       [[CONCAT_4:%.+]] = IE.Concat([[FQ_10]], [[FQ_12]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x16x96x352xf16>, tensor<1x16x96x352xf16> -> tensor<1x16x192x352xf16>
    // CHECK:       [[FQ_13:%.+]] = IE.FakeQuantize([[CONCAT_4]], {{[^:]+}}, {{[^:]+}}, {{[^:]+}}, {{[^:]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} :
    // CHECK-SAME:      tensor<1x16x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x16x192x352xf16>

    // CHECK:       return [[FQ_13]] : tensor<1x16x192x352xf16>
}
