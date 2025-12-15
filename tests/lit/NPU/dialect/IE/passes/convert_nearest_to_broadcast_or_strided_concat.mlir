//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nearest-to-broadcast-or-strided-concat %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW
func.func @ConvertNearestToStridedConcat_HW(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [12, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x20xf32>

    return %0 : tensor<1x128x12x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x20xf32>, tensor<1x128x6x20xf32> -> tensor<1x128x12x20xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x128x12x20xf32>
}

// -----

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW_ScalesMode
func.func @ConvertNearestToStridedConcat_HW_ScalesMode(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [12, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x20xf32>

    return %0 : tensor<1x128x12x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x20xf32>, tensor<1x128x6x20xf32> -> tensor<1x128x12x20xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x128x12x20xf32>
}

// -----

// CHECK-LABEL: @ConvertNearestToStridedConcat_H
func.func @ConvertNearestToStridedConcat_H(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x10xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 1.000000e+00],
         sizes_attr = [12, 10]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>

    return %0 : tensor<1x128x12x10xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_H:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>
    // CHECK: return [[CONCAT_H]] : tensor<1x128x12x10xf32>
}

// -----

// CHECK-LABEL: @ConvertNearestToStridedConcat_W
func.func @ConvertNearestToStridedConcat_W(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x6x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr =#IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00],
         sizes_attr = [6, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>

    return %0 : tensor<1x128x6x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_W:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: return [[CONCAT_W]] : tensor<1x128x6x20xf32>
}

// -----

// CHECK-LABEL: @ConvertNearestToStridedConcatFQPropagation
func.func @ConvertNearestToStridedConcatFQPropagation(%arg0: tensor<1x128x6x10xf16>) -> tensor<1x128x12x20xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x6x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x10xf16>

    %1 = IE.Interpolate(%0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [0.05328369140625, 0.0203399658203125], sizes_attr = [12, 20]} : tensor<1x128x6x10xf16> -> tensor<1x128x12x20xf16>

    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x12x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x12x20xf16>


    return %2 : tensor<1x128x12x20xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_0:%.*]] = IE.Concat
    // CHECK-SAME: {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>}
    // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[FQ_0]], [[FQ_0]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x6x20xf16>, tensor<1x128x6x20xf16> -> tensor<1x128x12x20xf16>
    // CHECK: [[CONCAT_OUT_FQ:%.*]] = IE.FakeQuantize
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolate4ToStridedConcat
func.func @ConvertNearestInterpolate4ToStridedConcat(%arg0: tensor<1x32x360x640xf32>) -> tensor<1x32x720x1280xf32> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 32, 720, 1280]} :
        tensor<1x32x360x640xf32> -> tensor<1x32x720x1280xf32>

    return %0 : tensor<1x32x720x1280xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x640xf32>, tensor<1x32x360x640xf32> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x1280xf32>, tensor<1x32x360x1280xf32> -> tensor<1x32x720x1280xf32>

    // CHECK: return [[CONCAT_2]] : tensor<1x32x720x1280xf32>
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolateToStridedConcat_CeilMode
func.func @ConvertNearestInterpolateToStridedConcat_CeilMode(%arg0: tensor<1x2x4x4xf16>) -> tensor<1x2x8x8xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <CEIL>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [8, 8]} :
        tensor<1x2x4x4xf16> -> tensor<1x2x8x8xf16>

    return %0 : tensor<1x2x8x8xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16> -> tensor<1x2x4x8xf16>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x4x8xf16>, tensor<1x2x4x8xf16> -> tensor<1x2x8x8xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 1] [1, 2, 8, 7] : tensor<1x2x8x8xf16> to tensor<1x2x8x7xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 7] [1, 2, 8, 1] : tensor<1x2x8x8xf16> to tensor<1x2x8x1xf16>
    // CHECK:   [[PAD_W:%.*]] = IE.Concat([[SLICE_1]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x8x7xf16>, tensor<1x2x8x1xf16> -> tensor<1x2x8x8xf16>
    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 1, 0] [1, 2, 7, 8] : tensor<1x2x8x8xf16> to tensor<1x2x7x8xf16>
    // CHECK:   [[SLICE_4:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 7, 0] [1, 2, 1, 8] : tensor<1x2x8x8xf16> to tensor<1x2x1x8xf16>
    // CHECK:   [[OUTPUT:%.*]] = IE.Concat([[SLICE_3]], [[SLICE_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x2x7x8xf16>, tensor<1x2x1x8xf16> -> tensor<1x2x8x8xf16>

    // CHECK: return [[OUTPUT]] : tensor<1x2x8x8xf16>
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolateToStridedConcat_RoundPreferFloorMode
func.func @ConvertNearestInterpolateToStridedConcat_RoundPreferFloorMode(%arg0: tensor<1x2x4x4xf16>) -> tensor<1x2x8x20xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [8, 20]} :
        tensor<1x2x4x4xf16> -> tensor<1x2x8x20xf16>

    return %0 : tensor<1x2x8x20xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0, %arg0, %arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 5 : i64>} : tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16> -> tensor<1x2x4x20xf16>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x4x20xf16>, tensor<1x2x4x20xf16> -> tensor<1x2x8x20xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 2] [1, 2, 8, 18] : tensor<1x2x8x20xf16> to tensor<1x2x8x18xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 19] [1, 2, 8, 1] : tensor<1x2x8x20xf16> to tensor<1x2x8x1xf16>
    // CHECK:   [[OUTPUT:%.*]] = IE.Concat([[SLICE_1]], [[SLICE_2]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x8x18xf16>, tensor<1x2x8x1xf16>, tensor<1x2x8x1xf16> -> tensor<1x2x8x20xf16>

    // CHECK: return [[OUTPUT]] : tensor<1x2x8x20xf16>
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolateToStridedConcat_RoundPreferCeilMode
func.func @ConvertNearestInterpolateToStridedConcat_RoundPreferCeilMode(%arg0: tensor<1x2x4x4xf16>) -> tensor<1x2x8x20xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <ROUND_PREFER_CEIL>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.3333300352096558, 1.3333300352096558], sizes_attr = [8, 20]} :
        tensor<1x2x4x4xf16> -> tensor<1x2x8x20xf16>

    return %0 : tensor<1x2x8x20xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0, %arg0, %arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 5 : i64>} : tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16>, tensor<1x2x4x4xf16> -> tensor<1x2x4x20xf16>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x4x20xf16>, tensor<1x2x4x20xf16> -> tensor<1x2x8x20xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 2] [1, 2, 8, 18] : tensor<1x2x8x20xf16> to tensor<1x2x8x18xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[CONCAT_2:%.+]] [0, 0, 0, 19] [1, 2, 8, 1] : tensor<1x2x8x20xf16> to tensor<1x2x8x1xf16>
    // CHECK:   [[PAD_W:%.*]] = IE.Concat([[SLICE_1]], [[SLICE_2]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x2x8x18xf16>, tensor<1x2x8x1xf16>, tensor<1x2x8x1xf16> -> tensor<1x2x8x20xf16>
    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 1, 0] [1, 2, 7, 20] : tensor<1x2x8x20xf16> to tensor<1x2x7x20xf16>
    // CHECK:   [[SLICE_4:%.+]] = IE.Slice [[PAD_W:%.+]] [0, 0, 7, 0] [1, 2, 1, 20] : tensor<1x2x8x20xf16> to tensor<1x2x1x20xf16>
    // CHECK:   [[OUTPUT:%.*]] = IE.Concat([[SLICE_3]], [[SLICE_4]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x2x7x20xf16>, tensor<1x2x1x20xf16> -> tensor<1x2x8x20xf16>

    // CHECK: return [[OUTPUT]] : tensor<1x2x8x20xf16>
}

// -----

// CHECK-LABEL: @ConvertNearestInterpolate4ToStridedConcatFQ
func.func @ConvertNearestInterpolate4ToStridedConcatFQ(%arg0: tensor<1x32x360x640xf32>) -> tensor<1x32x720x1280xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>

    %1 = IE.Interpolate(%0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 32, 720, 1280]} :
        tensor<1x32x360x640xf32> -> tensor<1x32x720x1280xf32>

    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>

    return %2 : tensor<1x32x720x1280xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK-DAG:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[FQ_1]], [[FQ_1]]) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x640xf32>, tensor<1x32x360x640xf32> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[FQ_2:%.*]] = IE.FakeQuantize([[CONCAT_1]], [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x360x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[FQ_2]], [[FQ_2]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x32x360x1280xf32>, tensor<1x32x360x1280xf32> -> tensor<1x32x720x1280xf32>
    // CHECK:   [[FQ_3:%.*]] = IE.FakeQuantize([[CONCAT_2]], [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>
    // CHECK:   return [[FQ_3]] : tensor<1x32x720x1280xf32>

}

// -----

// CHECK-LABEL: @NotConvertNearestForNonNTimesUpsampling
func.func @NotConvertNearestForNonNTimesUpsampling(%arg0: tensor<1x32x103x103xf32>) -> tensor<1x32x513x513xf32> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
        mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        scales_attr = [1.000000e+00, 1.000000e+00, 4.980600e+00, 4.980600e+00], sizes_attr = [1, 32, 513, 513]} :
        tensor<1x32x103x103xf32> -> tensor<1x32x513x513xf32>

    return %0 : tensor<1x32x513x513xf32>

    // CHECK: IE.Interpolate
}

// -----

// CHECK-LABEL: @NotConvertNearestScales
func.func @NotConvertNearestScales(%arg0: tensor<1x128x8x8xf16>) -> tensor<1x128x16x16xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>,
         coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 1, 1, 1]} : tensor<1x128x8x8xf16> -> tensor<1x128x16x16xf16>

    return %0 : tensor<1x128x16x16xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x8x8xf16>, tensor<1x128x8x8xf16> -> tensor<1x128x8x16xf16>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat(%0, %0) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x128x8x16xf16>, tensor<1x128x8x16xf16> -> tensor<1x128x16x16xf16>

    // CHECK: return [[CONCAT_2]] : tensor<1x128x16x16xf16>
}

// -----

// CHECK-LABEL: @NearestToBroadCastConverterWithSIZESMode
func.func @NearestToBroadCastConverterWithSIZESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CST:%.*]] = const.Declare tensor<4xsi32> = dense<[1, 96, 33, 33]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Broadcast(%arg0, [[CST]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x96x1x1xf32>, tensor<4xsi32> -> tensor<1x96x33x33xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x96x33x33xf32>
}

// -----

// CHECK-LABEL: @NearestToBroadCastConverterWithSCALESMode
func.func @NearestToBroadCastConverterWithSCALESMode(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CST:%.*]] = const.Declare tensor<4xsi32> = dense<[1, 96, 33, 33]> : tensor<4xsi64>, [#const.CastElemType<si32>]
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Broadcast(%arg0, [[CST]]) {mode = #IE.broadcast_type<NUMPY>} : tensor<1x96x1x1xf32>, tensor<4xsi32> -> tensor<1x96x33x33xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x96x33x33xf32>
}
