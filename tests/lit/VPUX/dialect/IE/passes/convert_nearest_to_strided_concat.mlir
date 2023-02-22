//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nearest-to-strided-concat %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW
func @ConvertNearestToStridedConcat_HW(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [12, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x20xf32>

    return %0 : tensor<1x128x12x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x20xf32>, tensor<1x128x6x20xf32> -> tensor<1x128x12x20xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x128x12x20xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcat_HW_ScalesMode
func @ConvertNearestToStridedConcat_HW_ScalesMode(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"},
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [12, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x20xf32>

    return %0 : tensor<1x128x12x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[CONCAT1]], [[CONCAT1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x20xf32>, tensor<1x128x6x20xf32> -> tensor<1x128x12x20xf32>
    // CHECK: return [[CONCAT_OUT]] : tensor<1x128x12x20xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcat_H
func @ConvertNearestToStridedConcat_H(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x12x10xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 1.000000e+00],
         sizes_attr = [12, 10]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>

    return %0 : tensor<1x128x12x10xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_H:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x12x10xf32>
    // CHECK: return [[CONCAT_H]] : tensor<1x128x12x10xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcat_W
func @ConvertNearestToStridedConcat_W(%arg0: tensor<1x128x6x10xf32>) -> tensor<1x128x6x20xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
         mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
         axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 2.000000e+00],
         sizes_attr = [6, 20]} :
         tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>

    return %0 : tensor<1x128x6x20xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_W:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x10xf32>, tensor<1x128x6x10xf32> -> tensor<1x128x6x20xf32>
    // CHECK: return [[CONCAT_W]] : tensor<1x128x6x20xf32>
}

// CHECK-LABEL: @ConvertNearestToStridedConcatFQPropagation
func @ConvertNearestToStridedConcatFQPropagation(%arg0: tensor<1x128x6x10xf16>) -> tensor<1x128x12x20xf16> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x128x6x10xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x6x10xf16>

    %1 = IE.Interpolate(%0) {attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [0.05328369140625, 0.0203399658203125], sizes_attr = [12, 20]} : tensor<1x128x6x10xf16> -> tensor<1x128x12x20xf16>

    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x128x12x20xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x128x12x20xf16>


    return %2 : tensor<1x128x12x20xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK: [[CONCAT_0:%.*]] = IE.Concat
    // CHECK-SAME: {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}}
    // CHECK: [[FQ_0:%.*]] = IE.FakeQuantize
    // CHECK: [[CONCAT_OUT:%.*]] = IE.Concat([[FQ_0]], [[FQ_0]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x128x6x20xf16>, tensor<1x128x6x20xf16> -> tensor<1x128x12x20xf16>
    // CHECK: [[CONCAT_OUT_FQ:%.*]] = IE.FakeQuantize
}

// CHECK-LABEL: @ConvertNearestInterpolate4ToStridedConcat
func @ConvertNearestInterpolate4ToStridedConcat(%arg0: tensor<1x32x360x640xf32>) -> tensor<1x32x720x1280xf32> {
    %0 = IE.Interpolate(%arg0) {
        attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
        mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"},
        axes_attr = [0, 1, 2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 32, 720, 1280]} :
        tensor<1x32x360x640xf32> -> tensor<1x32x720x1280xf32>

    return %0 : tensor<1x32x720x1280xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x360x640xf32>, tensor<1x32x360x640xf32> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x360x1280xf32>, tensor<1x32x360x1280xf32> -> tensor<1x32x720x1280xf32>

    // CHECK: return [[CONCAT_2]] : tensor<1x32x720x1280xf32>
}

// CHECK-LABEL: @ConvertNearestInterpolate4ToStridedConcatFQ
func @ConvertNearestInterpolate4ToStridedConcatFQ(%arg0: tensor<1x32x360x640xf32>) -> tensor<1x32x720x1280xf32> {
    %cst_0 = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    %cst_1 = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>

    %0 = IE.FakeQuantize(%arg0, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>

    %1 = IE.Interpolate(%0) {
        attr = {antialias = false, coord_mode = "ASYMMETRIC", cube_coeff = -7.500000e-01 : f64,
        mode = "NEAREST", nearest_mode = "FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"},
        axes_attr = [0, 1, 2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], sizes_attr = [1, 32, 720, 1280]} :
        tensor<1x32x360x640xf32> -> tensor<1x32x720x1280xf32>

    %2 = IE.FakeQuantize(%1, %cst_0, %cst_1, %cst_0, %cst_1) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>

    return %2 : tensor<1x32x720x1280xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CST:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:   [[CST_1:%.*]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:   [[FQ_1:%.*]] = IE.FakeQuantize(%arg0, [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x360x640xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x640xf32>
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat([[FQ_1]], [[FQ_1]]) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x360x640xf32>, tensor<1x32x360x640xf32> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[FQ_2:%.*]] = IE.FakeQuantize([[CONCAT_1]], [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x360x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x360x1280xf32>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[FQ_2]], [[FQ_2]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x360x1280xf32>, tensor<1x32x360x1280xf32> -> tensor<1x32x720x1280xf32>
    // CHECK:   [[FQ_3:%.*]] = IE.FakeQuantize([[CONCAT_2]], [[CST]], [[CST_1]], [[CST]], [[CST_1]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x720x1280xf32>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x720x1280xf32>
    // CHECK:   return [[FQ_3]] : tensor<1x32x720x1280xf32>

}
