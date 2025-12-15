//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nearest-to-broadcast-or-strided-concat %s | FileCheck %s
// REQUIRES: arch-NPU50XX
// COM: F8 is only supported on NPU50+, no need to run these tests on all arches.

// CHECK-LABEL: @ConvertNearestToStridedConcatFQuantPropagationF8E4M3FN
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x128x6x10xf16>
func.func @ConvertNearestToStridedConcatFQuantPropagationF8E4M3FN(%input: tensor<1x128x6x10xf16>) -> tensor<1x128x12x20xf16> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
        } : tensor<1x128x6x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x10xf16>

    %1 = IE.Interpolate(%0) {
            attr = #IE.Interpolate<
                antialias = false, coord_mode = <ASYMMETRIC>,
                cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
                pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            scales_attr = [0.05328369140625, 0.0203399658203125], sizes_attr = [12, 20]
        } : tensor<1x128x6x10xf16> -> tensor<1x128x12x20xf16>

    %2 = IE.FakeQuantize(%1, %low, %high, %low, %high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN
        } : tensor<1x128x12x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x12x20xf16>

    return %2 : tensor<1x128x12x20xf16>

    // CHECK-NOT: IE.Interpolate

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-4.480000e+02> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.480000e+02> : tensor<1x1x1x1xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x128x6x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x10xf16>
    // CHECK:    [[CONCAT_0:%.+]] = IE.Concat([[FQ_0]], [[FQ_0]])
    // CHECK-SAME:  {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x10xf16>, tensor<1x128x6x10xf16> -> tensor<1x128x6x20xf16>

    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[CONCAT_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x128x6x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x20xf16>
    // CHECK:    [[CONCAT_1:%.+]] = IE.Concat([[FQ_1]], [[FQ_1]])
    // CHECK-SAME:  {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x20xf16>, tensor<1x128x6x20xf16> -> tensor<1x128x12x20xf16>

    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[CONCAT_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E4M3FN} : tensor<1x128x12x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x12x20xf16>

    // CHECK:    return [[FQ_2]] : tensor<1x128x12x20xf16>
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolate4ToStridedConcatFQuantF8E5M2
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x32x360x640xf32>
func.func @ConvertNearestInterpolate4ToStridedConcatFQuantF8E5M2(%input: tensor<1x32x360x640xf32>) -> tensor<1x32x720x1280xf32> {
    %low = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    %high = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%input, %low, %high, %low, %high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
        } : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>

    %1 = IE.Interpolate(%0) {
            attr = #IE.Interpolate<
                antialias = false, coord_mode = <ASYMMETRIC>,
                cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>,
                pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
            axes_attr = [0, 1, 2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0>,
            scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 32, 720, 1280]
        } : tensor<1x32x360x640xf32> -> tensor<1x32x720x1280xf32>

    %2 = IE.FakeQuantize(%1, %low, %high, %low, %high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2
        } : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>

    return %2 : tensor<1x32x720x1280xf32>

    // CHECK-DAG:    [[LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-5.734400e+04> : tensor<1x1x1x1xf16>
    // CHECK-DAG:    [[HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<5.734400e+04> : tensor<1x1x1x1xf16>

    // CHECK:    [[FQ_0:%.+]] = IE.FakeQuantize([[INPUT]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>
    // CHECK:    [[CONCAT_0:%.+]] = IE.Concat([[FQ_0]], [[FQ_0]])
    // CHECK-SAME:  {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x640xf32>, tensor<1x32x360x640xf32> -> tensor<1x32x360x1280xf32>

    // CHECK:    [[FQ_1:%.+]] = IE.FakeQuantize([[CONCAT_0]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x360x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x1280xf32>
    // CHECK:    [[CONCAT_1:%.+]] = IE.Concat([[FQ_1]], [[FQ_1]])
    // CHECK-SAME:  {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x1280xf32>, tensor<1x32x360x1280xf32> -> tensor<1x32x720x1280xf32>

    // CHECK:    [[FQ_2:%.+]] = IE.FakeQuantize([[CONCAT_1]], [[LOW]], [[HIGH]], [[LOW]], [[HIGH]])
    // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, low_fp_type = f8E5M2} : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>

    // CHECK:    return [[FQ_2]] : tensor<1x32x720x1280xf32>
}
