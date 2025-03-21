//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-parallel-slices-to-gather %s | FileCheck %s
// REQUIRES: arch-NPU40XX
// CHECK-LABEL: @ConvertParallelSliceBranchesToGather
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: tensor<6x9216x1x1xf16>
func.func @ConvertParallelSliceBranchesToGather(
                %arg0: tensor<8x12288x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>,
                %arg2 : tensor<8x12288x1x1xf16>,
                %arg3 : tensor<6x9216x1x1xf16>) -> tensor<1x30x1x1536xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %1 = IE.Slice %0 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %2 = IE.Slice %0 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %3 = IE.Slice %0 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %4 = IE.Slice %0 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %5 = IE.Slice %0 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %6 = IE.Slice %0 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %7 = IE.Slice %0 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %8 = IE.Slice %0 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 1
    %9 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %10 = IE.Slice %9 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %11 = IE.Slice %9 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %12 = IE.Slice %9 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %13 = IE.Slice %9 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %14 = IE.Slice %9 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %15 = IE.Slice %9 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %16 = IE.Slice %9 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %17 = IE.Slice %9 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 2
    %18 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %19 = IE.Slice %18 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %20 = IE.Slice %18 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %21 = IE.Slice %18 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %22 = IE.Slice %18 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %23 = IE.Slice %18 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %24 = IE.Slice %18 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %25 = IE.Slice %18 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %26 = IE.Slice %18 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 3
    %27 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>
    %28 = IE.Slice %27 [0, 0] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %29 = IE.Slice %27 [1, 1536] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %30 = IE.Slice %27 [2, 3072] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %31 = IE.Slice %27 [3, 4608] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %32 = IE.Slice %27 [4, 6144] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %33 = IE.Slice %27 [5, 7680] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>

    // concat
    %34 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %10, %11, %12, %13, %14, %15, %16, %17, %19, %20, %21, %22, %23, %24, %25, %26, %28, %29, %30, %31, %32, %33)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
                               [13, 0], [14, 0], [15, 0], [16, 0], [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [23, 0], [24, 0],
                               [25, 0], [26, 0], [27, 0], [28, 0], [29, 0]]
            } : tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<30x1536xf16>
    %35 = IE.AffineReshape(%34) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 30, 1, 1536]} : tensor<30x1536xf16> -> tensor<1x30x1x1536xf16>

    return %35 : tensor<1x30x1x1536xf16>

    // CHECK-DAG:       [[INDICES_1x6:%.+]] = const.Declare tensor<1x6xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 7, 14, 21, 28, 35]]> : tensor<1x6xsi32>
    // CHECK-DAG:       [[INDICES_1x8:%.+]] = const.Declare tensor<1x8xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 9, 18, 27, 36, 45, 54, 63]]> : tensor<1x8xsi32>

    // CHECK:           [[SOURCE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_2:%.+]] = IE.AffineReshape([[INPUT_2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_3:%.+]] = IE.AffineReshape([[INPUT_3]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>

    // CHECK:           [[SOURCE_RESHAPE_0:%.+]] = IE.Reshape([[SOURCE_0]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_0:%.+]] = IE.Gather([[SOURCE_RESHAPE_0]], [[INDICES_1x8]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x8xsi32> -> tensor<1x8x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_0:%.+]] = IE.Reshape([[GATHER_0]]) {shape_value = [8, 1536]} : tensor<1x8x1536xf16> -> tensor<8x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_1:%.+]] = IE.Reshape([[SOURCE_1]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_1:%.+]] = IE.Gather([[SOURCE_RESHAPE_1]], [[INDICES_1x8]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x8xsi32> -> tensor<1x8x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_1:%.+]] = IE.Reshape([[GATHER_1]]) {shape_value = [8, 1536]} : tensor<1x8x1536xf16> -> tensor<8x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_2:%.+]] = IE.Reshape([[SOURCE_2]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_2:%.+]] = IE.Gather([[SOURCE_RESHAPE_2]], [[INDICES_1x8]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x8xsi32> -> tensor<1x8x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_2:%.+]] = IE.Reshape([[GATHER_2]]) {shape_value = [8, 1536]} : tensor<1x8x1536xf16> -> tensor<8x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_3:%.+]] = IE.Reshape([[SOURCE_3]]) {shape_value = [36, 1536]} : tensor<6x9216xf16> -> tensor<36x1536xf16>
    // CHECK:           [[GATHER_3:%.+]] = IE.Gather([[SOURCE_RESHAPE_3]], [[INDICES_1x6]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<36x1536xf16>, tensor<1x6xsi32> -> tensor<1x6x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_3:%.+]] = IE.Reshape([[GATHER_3]]) {shape_value = [6, 1536]} : tensor<1x6x1536xf16> -> tensor<6x1536xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[GATHER_RESHAPE_0]], [[GATHER_RESHAPE_1]], [[GATHER_RESHAPE_2]], [[GATHER_RESHAPE_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<8x1536xf16>, tensor<8x1536xf16>, tensor<8x1536xf16>, tensor<6x1536xf16> -> tensor<30x1536xf16>
    // CHECK:           [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 30, 1, 1536]} : tensor<30x1536xf16> -> tensor<1x30x1x1536xf16>

    // CHECK:           return [[OUT_RESHAPE]] : tensor<1x30x1x1536xf16>
}

// -----

// CHECK-LABEL: @NotConvertIfAllGroupsContainOnlyOneSliceOp
func.func @NotConvertIfAllGroupsContainOnlyOneSliceOp(
                %arg0: tensor<8x12288x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>,
                %arg2 : tensor<8x12288x1x1xf16>,
                %arg3 : tensor<6x9216x1x1xf16>) -> tensor<1x4x1x1536xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %1 = IE.Slice %0 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 1
    %2 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %3 = IE.Slice %2 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 2
    %4 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %5 = IE.Slice %4 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 3
    %6 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>
    %7 = IE.Slice %6 [0, 0] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>

    // concat
    %8 = IE.Concat(%1, %3, %5, %7)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0]]
            } : tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<4x1536xf16>
    %9 = IE.AffineReshape(%8) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 4, 1, 1536]} : tensor<4x1536xf16> -> tensor<1x4x1x1536xf16>

    return %9 : tensor<1x4x1x1536xf16>

    // CHECK-NOT:       IE.Gather
}

// -----

// CHECK-LABEL: @ConvertParallelSliceBranchesToGatherLessBranches
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: tensor<6x9216x1x1xf16>
func.func @ConvertParallelSliceBranchesToGatherLessBranches(
                %arg0: tensor<8x12288x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>,
                %arg2 : tensor<8x12288x1x1xf16>,
                %arg3 : tensor<6x9216x1x1xf16>) -> tensor<1x8x1x1536xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %1 = IE.Slice %0 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %2 = IE.Slice %0 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 1
    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %4 = IE.Slice %3 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %5 = IE.Slice %3 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 2
    %6 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %7 = IE.Slice %6 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %8 = IE.Slice %6 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 3
    %9 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>
    %10 = IE.Slice %9 [2, 3072] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>
    %11 = IE.Slice %9 [3, 4608] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>

    // concat
    %12 = IE.Concat(%1, %2, %4, %5, %7, %8, %10, %11)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]]} :
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<8x1536xf16>
    %13 = IE.AffineReshape(%12) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 8, 1, 1536]} : tensor<8x1536xf16> -> tensor<1x8x1x1536xf16>

    return %13 : tensor<1x8x1x1536xf16>

    // CHECK-DAG:       [[INDICES_3:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[14, 21]]> : tensor<1x2xsi32>
    // CHECK-DAG:       [[INDICES_2:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 9]]> : tensor<1x2xsi32>
    // CHECK-DAG:       [[INDICES_1:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[9, 27]]> : tensor<1x2xsi32>
    // CHECK-DAG:       [[INDICES_0:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 18]]> : tensor<1x2xsi32>

    // CHECK:           [[SOURCE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_2:%.+]] = IE.AffineReshape([[INPUT_2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_3:%.+]] = IE.AffineReshape([[INPUT_3]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>

    // CHECK:           [[SOURCE_RESHAPE_0:%.+]] = IE.Reshape([[SOURCE_0]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_0:%.+]] = IE.Gather([[SOURCE_RESHAPE_0]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_0:%.+]] = IE.Reshape([[GATHER_0]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_1:%.+]] = IE.Reshape([[SOURCE_1]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_1:%.+]] = IE.Gather([[SOURCE_RESHAPE_1]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_1:%.+]] = IE.Reshape([[GATHER_1]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_2:%.+]] = IE.Reshape([[SOURCE_2]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_2:%.+]] = IE.Gather([[SOURCE_RESHAPE_2]], [[INDICES_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_2:%.+]] = IE.Reshape([[GATHER_2]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_3:%.+]] = IE.Reshape([[SOURCE_3]]) {shape_value = [36, 1536]} : tensor<6x9216xf16> -> tensor<36x1536xf16>
    // CHECK:           [[GATHER_3:%.+]] = IE.Gather([[SOURCE_RESHAPE_3]], [[INDICES_3]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<36x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_3:%.+]] = IE.Reshape([[GATHER_3]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[GATHER_RESHAPE_0]], [[GATHER_RESHAPE_1]], [[GATHER_RESHAPE_2]], [[GATHER_RESHAPE_3]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x1536xf16>, tensor<2x1536xf16>, tensor<2x1536xf16>, tensor<2x1536xf16> -> tensor<8x1536xf16>
    // CHECK:           [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 8, 1, 1536]} : tensor<8x1536xf16> -> tensor<1x8x1x1536xf16>

    // CHECK:           return [[OUT_RESHAPE]] : tensor<1x8x1x1536xf16>
}

// -----

// CHECK-LABEL: @ConvertParallelSliceBranchesToGatherWithSingleSliceOpInGroup
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<8x12288x1x1xf16>
// CHECK-SAME:      [[INPUT_3:%arg[0-9]]]: tensor<6x9216x1x1xf16>
func.func @ConvertParallelSliceBranchesToGatherWithSingleSliceOpInGroup(
                %arg0: tensor<8x12288x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>,
                %arg2 : tensor<8x12288x1x1xf16>,
                %arg3 : tensor<6x9216x1x1xf16>) -> tensor<1x7x1x1536xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %1 = IE.Slice %0 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %2 = IE.Slice %0 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 1
    %3 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %4 = IE.Slice %3 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %5 = IE.Slice %3 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 2
    %6 = IE.AffineReshape(%arg2) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %7 = IE.Slice %6 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %8 = IE.Slice %6 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 3
    %9 = IE.AffineReshape(%arg3) {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>
    %10 = IE.Slice %9 [2, 3072] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>

    // concat
    %11 = IE.Concat(%1, %2, %4, %5, %7, %8, %10)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]} :
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16> -> tensor<7x1536xf16>
    %12 = IE.AffineReshape(%11) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 7, 1, 1536]} : tensor<7x1536xf16> -> tensor<1x7x1x1536xf16>

    return %12 : tensor<1x7x1x1536xf16>

    // CHECK-DAG:       [[INDICES_2:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 9]]> : tensor<1x2xsi32>
    // CHECK-DAG:       [[INDICES_1:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[9, 27]]> : tensor<1x2xsi32>
    // CHECK-DAG:       [[INDICES_0:%.+]] = const.Declare tensor<1x2xsi32> =
    // CHECK-SAME{LITERAL}:     dense<[[0, 18]]> : tensor<1x2xsi32>

    // CHECK:           [[SOURCE_0:%.+]] = IE.AffineReshape([[INPUT_0]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_1:%.+]] = IE.AffineReshape([[INPUT_1]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_2:%.+]] = IE.AffineReshape([[INPUT_2]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // CHECK:           [[SOURCE_3:%.+]] = IE.AffineReshape([[INPUT_3]])
    // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1], [1], [1]], shape_value = [6, 9216]} : tensor<6x9216x1x1xf16> -> tensor<6x9216xf16>

    // CHECK:           [[SLICE:%.+]] = IE.Slice [[SOURCE_3]] [2, 3072] [1, 1536] : tensor<6x9216xf16> to tensor<1x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_0:%.+]] = IE.Reshape([[SOURCE_0]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_0:%.+]] = IE.Gather([[SOURCE_RESHAPE_0]], [[INDICES_0]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_0:%.+]] = IE.Reshape([[GATHER_0]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_1:%.+]] = IE.Reshape([[SOURCE_1]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_1:%.+]] = IE.Gather([[SOURCE_RESHAPE_1]], [[INDICES_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_1:%.+]] = IE.Reshape([[GATHER_1]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[SOURCE_RESHAPE_2:%.+]] = IE.Reshape([[SOURCE_2]]) {shape_value = [64, 1536]} : tensor<8x12288xf16> -> tensor<64x1536xf16>
    // CHECK:           [[GATHER_2:%.+]] = IE.Gather([[SOURCE_RESHAPE_2]], [[INDICES_2]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<64x1536xf16>, tensor<1x2xsi32> -> tensor<1x2x1536xf16>
    // CHECK:           [[GATHER_RESHAPE_2:%.+]] = IE.Reshape([[GATHER_2]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf16> -> tensor<2x1536xf16>

    // CHECK:           [[CONCAT:%.+]] = IE.Concat([[GATHER_RESHAPE_0]], [[GATHER_RESHAPE_1]], [[GATHER_RESHAPE_2]], [[SLICE]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<2x1536xf16>, tensor<2x1536xf16>, tensor<2x1536xf16>, tensor<1x1536xf16> -> tensor<7x1536xf16>
    // CHECK:           [[OUT_RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK-SAME{LITERAL}      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 8, 1, 1536]} : tensor<7x1536xf16> -> tensor<1x7x1x1536xf16>

    // CHECK:           return [[OUT_RESHAPE]] : tensor<1x7x1x1536xf16>
}

// -----

// CHECK-LABEL: @NotConvertDueToInvalidSliceOffsets
func.func @NotConvertDueToInvalidSliceOffsets(
                %arg0: tensor<8x12288x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>) -> tensor<1x16x1x1536xf16> {
    // group 0
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // Slice has invalid offsets [0, 5]
    %1 = IE.Slice %0 [0, 5] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %2 = IE.Slice %0 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %3 = IE.Slice %0 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %4 = IE.Slice %0 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %5 = IE.Slice %0 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %6 = IE.Slice %0 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %7 = IE.Slice %0 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %8 = IE.Slice %0 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // group 1
    %9 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    // Slice has invalid offsets [0, 10]
    %10 = IE.Slice %9 [0, 10] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %11 = IE.Slice %9 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %12 = IE.Slice %9 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %13 = IE.Slice %9 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %14 = IE.Slice %9 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %15 = IE.Slice %9 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %16 = IE.Slice %9 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %17 = IE.Slice %9 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // concat
    %18 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %10, %11, %12, %13, %14, %15, %16, %17)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
                               [13, 0], [14, 0], [15, 0]]
            } : tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<16x1536xf16>
    %19 = IE.AffineReshape(%18) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 1, 1536]} : tensor<16x1536xf16> -> tensor<1x16x1x1536xf16>

    return %19 : tensor<1x16x1x1536xf16>

    // CHECK-NOT:       IE.Gather
}

// -----

// CHECK-LABEL: @NotConvertDueToInvalidSourceShape
func.func @NotConvertDueToInvalidSourceShape(
                %arg0: tensor<8x12290x1x1xf16>,
                %arg1 : tensor<8x12288x1x1xf16>) -> tensor<1x16x1x1536xf16> {
    // group 0: source shape 8x12290, 12290 can't be divided by elementSize 1536
    %0 = IE.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12290]} : tensor<8x12290x1x1xf16> -> tensor<8x12290xf16>
    %1 = IE.Slice %0 [0, 0] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %2 = IE.Slice %0 [1, 1536] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %3 = IE.Slice %0 [2, 3072] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %4 = IE.Slice %0 [3, 4608] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %5 = IE.Slice %0 [4, 6144] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %6 = IE.Slice %0 [5, 7680] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %7 = IE.Slice %0 [6, 9216] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>
    %8 = IE.Slice %0 [7, 10752] [1, 1536] : tensor<8x12290xf16> to tensor<1x1536xf16>

    // group 1
    %9 = IE.AffineReshape(%arg1) {dim_mapping = [[0], [1], [1], [1]], shape_value = [8, 12288]} : tensor<8x12288x1x1xf16> -> tensor<8x12288xf16>
    %10 = IE.Slice %9 [0, 0] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %11 = IE.Slice %9 [1, 1536] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %12 = IE.Slice %9 [2, 3072] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %13 = IE.Slice %9 [3, 4608] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %14 = IE.Slice %9 [4, 6144] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %15 = IE.Slice %9 [5, 7680] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %16 = IE.Slice %9 [6, 9216] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>
    %17 = IE.Slice %9 [7, 10752] [1, 1536] : tensor<8x12288xf16> to tensor<1x1536xf16>

    // concat
    %18 = IE.Concat(%1, %2, %3, %4, %5, %6, %7, %8, %10, %11, %12, %13, %14, %15, %16, %17)
            {static_offsets = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
                               [13, 0], [14, 0], [15, 0]]
            } : tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>,
                tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16>, tensor<1x1536xf16> -> tensor<16x1536xf16>
    %19 = IE.AffineReshape(%18) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 16, 1, 1536]} : tensor<16x1536xf16> -> tensor<1x16x1x1536xf16>

    return %19 : tensor<1x16x1x1536xf16>

    // CHECK-NOT:       IE.Gather
}
