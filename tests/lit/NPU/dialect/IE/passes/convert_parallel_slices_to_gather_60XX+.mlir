//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-parallel-slices-to-gather %s | FileCheck %s
// REQUIRES: platform-NPU6010

// CHECK-LABEL: @ConvertParallelSliceBranchesToGatherWithLargeDimSize
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<8x61440x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<4x30720x1x1xf16>
func.func @ConvertParallelSliceBranchesToGatherWithLargeDimSize(
                %arg0: tensor<8x61440x1x1xf16>,
                %arg1 : tensor<4x30720x1x1xf16>) -> tensor<1x12x1x7680xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 61440]} : tensor<8x61440x1x1xf16> -> tensor<8x61440xf16>
    %1 = IE.Slice %0 [0, 0] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %2 = IE.Slice %0 [1, 7680] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %3 = IE.Slice %0 [2, 15360] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %4 = IE.Slice %0 [3, 23040] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %5 = IE.Slice %0 [4, 30720] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %6 = IE.Slice %0 [5, 38400] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %7 = IE.Slice %0 [6, 46080] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>
    %8 = IE.Slice %0 [7, 53760] [1, 7680] : tensor<8x61440xf16> to tensor<1x7680xf16>

    // group 1
    %9 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [4, 30720]} : tensor<4x30720x1x1xf16> -> tensor<4x30720xf16>
    %10 = IE.Slice %9 [0, 0] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %11 = IE.Slice %9 [1, 7680] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %12 = IE.Slice %9 [2, 15360] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>
    %13 = IE.Slice %9 [3, 23040] [1, 7680] : tensor<4x30720xf16> to tensor<1x7680xf16>

    // concat
    %14 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %10, %11, %12, %13)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]]
            } : tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>,
                tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16>, tensor<1x7680xf16> -> tensor<12x7680xf16>
    %15 = IE.AffineReshape(%14) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 12, 1, 7680]} : tensor<12x7680xf16> -> tensor<1x12x1x7680xf16>

    return %15 : tensor<1x12x1x7680xf16>


    // CHECK:           [[SOURCE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 61440]} : tensor<8x61440x1x1xf16> -> tensor<8x61440xf16>
    // CHECK:           [[SOURCE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [4, 30720]} : tensor<4x30720x1x1xf16> -> tensor<4x30720xf16>

    // CHECK:           [[SOURCE_RESHAPE_0:%.+]] = IE.Reshape([[SOURCE_0]]) {shape_value = [64, 7680]} : tensor<8x61440xf16> -> tensor<64x7680xf16>
    // CHECK:           [[INDICES_1x8:%.+]] = const.Declare tensor<1x8xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 9, 18, 27, 36, 45, 54, 63]]> : tensor<1x8xsi32>
    // CHECK:           [[GATHER_0:%.+]] = IE.Gather([[SOURCE_RESHAPE_0]], [[INDICES_1x8]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x7680xf16>, tensor<1x8xsi32> -> tensor<1x8x7680xf16>
    // CHECK:           [[GATHER_RESHAPE_0:%.+]] = IE.Reshape([[GATHER_0]]) {shape_value = [8, 7680]} : tensor<1x8x7680xf16> -> tensor<8x7680xf16>

    // CHECK:           [[SOURCE_RESHAPE_1:%.+]] = IE.Reshape([[SOURCE_1]]) {shape_value = [16, 7680]} : tensor<4x30720xf16> -> tensor<16x7680xf16>
    // CHECK:           [[INDICES_1x4:%.+]] = const.Declare tensor<1x4xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 5, 10, 15]]> : tensor<1x4xsi32>
    // CHECK:           [[GATHER_1:%.+]] = IE.Gather([[SOURCE_RESHAPE_1]], [[INDICES_1x4]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<16x7680xf16>, tensor<1x4xsi32> -> tensor<1x4x7680xf16>
    // CHECK:           [[GATHER_RESHAPE_1:%.+]] = IE.Reshape([[GATHER_1]]) {shape_value = [4, 7680]} : tensor<1x4x7680xf16> -> tensor<4x7680xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[GATHER_RESHAPE_0]], [[GATHER_RESHAPE_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<8x7680xf16>, tensor<4x7680xf16> -> tensor<12x7680xf16>
    // CHECK:           [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 12, 1, 7680]} : tensor<12x7680xf16> -> tensor<1x12x1x7680xf16>

    // CHECK:           return [[OUT_RESHAPE]] : tensor<1x12x1x7680xf16>
}
