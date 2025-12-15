//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nearest-to-broadcast-or-strided-concat="interpolate-as-se-op=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @ConvertNearestWithSmallChannelAndSmallSpatialSize
func.func @ConvertNearestWithSmallChannelAndSmallSpatialSize(%arg0: tensor<1x4x160x160xf32>) -> tensor<1x4x640x640xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [4.000000e+00, 4.000000e+00],
         sizes_attr = [640, 640]} :
         tensor<1x4x160x160xf32> -> tensor<1x4x640x640xf32>

    return %0 : tensor<1x4x640x640xf32>

    // CHECK-NOT: IE.Interpolate
    // CHECK:   [[CONCAT_1:%.*]] = IE.Concat(%arg0, %arg0, %arg0, %arg0) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x4x160x160xf32>, tensor<1x4x160x160xf32>, tensor<1x4x160x160xf32>, tensor<1x4x160x160xf32> -> tensor<1x4x160x640xf32>
    // CHECK:   [[CONCAT_2:%.*]] = IE.Concat([[CONCAT_1]], [[CONCAT_1]], [[CONCAT_1]], [[CONCAT_1]]) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 4 : i64>} : tensor<1x4x160x640xf32>, tensor<1x4x160x640xf32>, tensor<1x4x160x640xf32>, tensor<1x4x160x640xf32> -> tensor<1x4x640x640xf32>
    // CHECK:   return [[CONCAT_2]] : tensor<1x4x640x640xf32>
}

// -----

// CHECK-LABEL: @DontConvertNearestWithSmallChannelAndLargeSpatialSize
func.func @DontConvertNearestWithSmallChannelAndLargeSpatialSize(%arg0: tensor<1x3x736x1280xf32>) -> tensor<1x3x1472x2560xf32> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64,
         mode = <NEAREST>, nearest_mode = <FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
         axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00],
         sizes_attr = [1472, 2560]} :
         tensor<1x3x736x1280xf32> -> tensor<1x3x1472x2560xf32>

    return %0 : tensor<1x3x1472x2560xf32>

    // CHECK-NOT: IE.Concat
    // CHECK: [[OUT:%.*]] = IE.Interpolate(%arg0)
    // CHECK: return [[OUT]] : tensor<1x3x1472x2560xf32>
}
