//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ExpandInputRank1D
// CHECK-SAME:  ([[ARG:%.+]]: tensor<8xsi32>)
func.func @ExpandInputRank1D(%arg0: tensor<8xsi32>) -> tensor<4x8xf16> {
    %0 = IE.OneHot(%arg0) {
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        depth_attr = 4 : i64,
        on_value_attr = 1.000000e+00 : f64,
        off_value_attr = 0.000000e+00 : f64,
        axis_attr = 0 : i64,
        mode = #IE.one_hot_mode<IGNORE_NEGATIVE>,
        outputType = f16
    } : tensor<8xsi32> -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:      shape_value = [1, 8]
    // CHECK-SAME:      tensor<8xsi32> -> tensor<1x8xsi32>
    // CHECK:       [[OUT:%.+]] = IE.OneHot([[RESHAPE]])
    // CHECK-SAME:      axis_attr = 0 : i64
    // CHECK-SAME:      depth_attr = 4 : i64
    // CHECK-SAME:      rank_expanded = true
    // CHECK-SAME:      tensor<1x8xsi32> -> tensor<4x8xf16>
    // CHECK:       return [[OUT]] : tensor<4x8xf16>
}

// -----

// CHECK-LABEL: @ExpandInputRank2D
// CHECK-SAME:  ([[ARG:%.+]]: tensor<3x8xsi32>)
func.func @ExpandInputRank2D(%arg0: tensor<3x8xsi32>) -> tensor<3x4x8xf16> {
    %0 = IE.OneHot(%arg0) {
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        depth_attr = 4 : i64,
        on_value_attr = 1.000000e+00 : f64,
        off_value_attr = 0.000000e+00 : f64,
        axis_attr = 1 : i64,
        mode = #IE.one_hot_mode<IGNORE_NEGATIVE>,
        outputType = f16
    } : tensor<3x8xsi32> -> tensor<3x4x8xf16>
    return %0 : tensor<3x4x8xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:      shape_value = [3, 1, 8]
    // CHECK-SAME:      tensor<3x8xsi32> -> tensor<3x1x8xsi32>
    // CHECK:       [[OUT:%.+]] = IE.OneHot([[RESHAPE]])
    // CHECK-SAME:      axis_attr = 1 : i64
    // CHECK-SAME:      depth_attr = 4 : i64
    // CHECK-SAME:      rank_expanded = true
    // CHECK-SAME:      tensor<3x1x8xsi32> -> tensor<3x4x8xf16>
    // CHECK:       return [[OUT]] : tensor<3x4x8xf16>
}

// -----

// CHECK-LABEL: @NormalizeNegativeAxis
// CHECK-SAME:  ([[ARG:%.+]]: tensor<3x8xsi32>)
func.func @NormalizeNegativeAxis(%arg0: tensor<3x8xsi32>) -> tensor<3x4x8xf16> {
    %0 = IE.OneHot(%arg0) {
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        depth_attr = 4 : i64,
        on_value_attr = 1.000000e+00 : f64,
        off_value_attr = 0.000000e+00 : f64,
        axis_attr = -2 : i64,
        mode = #IE.one_hot_mode<IGNORE_NEGATIVE>,
        outputType = f16
    } : tensor<3x8xsi32> -> tensor<3x4x8xf16>
    return %0 : tensor<3x4x8xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[ARG]])
    // CHECK-SAME:      shape_value = [3, 1, 8]
    // CHECK-SAME:      tensor<3x8xsi32> -> tensor<3x1x8xsi32>
    // CHECK:       [[OUT:%.+]] = IE.OneHot([[RESHAPE]])
    // CHECK-SAME:      axis_attr = 1 : i64
    // CHECK-SAME:      depth_attr = 4 : i64
    // CHECK-SAME:      rank_expanded = true
    // CHECK-SAME:      tensor<3x1x8xsi32> -> tensor<3x4x8xf16>
    // CHECK:       return [[OUT]] : tensor<3x4x8xf16>
}

// -----

// CHECK-LABEL: @NoExpandWhenAlreadyExpanded
// CHECK-SAME:  ([[ARG:%.+]]: tensor<3x1x8xsi32>)
func.func @NoExpandWhenAlreadyExpanded(%arg0: tensor<3x1x8xsi32>) -> tensor<3x4x8xf16> {
    %0 = IE.OneHot(%arg0) {
        operandSegmentSizes = array<i32: 1, 0, 0, 0>,
        depth_attr = 4 : i64,
        on_value_attr = 1.000000e+00 : f64,
        off_value_attr = 0.000000e+00 : f64,
        axis_attr = 1 : i64,
        mode = #IE.one_hot_mode<IGNORE_NEGATIVE>,
        rank_expanded = true,
        outputType = f16
    } : tensor<3x1x8xsi32> -> tensor<3x4x8xf16>
    return %0 : tensor<3x4x8xf16>

    // CHECK-NOT:   IE.Reshape
    // CHECK:       [[OUT:%.+]] = IE.OneHot([[ARG]])
    // CHECK-SAME:      axis_attr = 1 : i64
    // CHECK-SAME:      depth_attr = 4 : i64
    // CHECK-SAME:      rank_expanded = true
    // CHECK:       return [[OUT]] : tensor<3x4x8xf16>
}
