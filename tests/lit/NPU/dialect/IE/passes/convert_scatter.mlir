//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-scatter --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertScatterNDUpdateToStridedConcat
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1x15xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x1x5xf16>
func.func @ConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,12]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_1:%.+]] = IE.StridedSlice([[ARG_0]]) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[SLICE_2:%.+]] = IE.StridedSlice([[ARG_0]]) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 2], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_1]], [[SLICE_1]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 4 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]] : tensor<1x1x1x1x15xf16>
}

// -----

// The last dim value is [0,0,0,0,11], non-uniform stride so StridedConcat cannot handle it,
// but the select pattern converts it to Slice+Concat selecting positions [0,3,6,9,11] from dim4.
// CHECK-LABEL: @ConvertNonUniformStrideScatterNDUpdateViaSelect
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1x15xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x1x5xf16>
func.func @ConvertNonUniformStrideScatterNDUpdateViaSelect(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,11]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_U0:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 0] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_D0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 1] [1, 1, 1, 1, 2]
    // CHECK: [[SLICE_U1:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 1] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_D1:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 4] [1, 1, 1, 1, 2]
    // CHECK: [[SLICE_U2:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 2] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_D2:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 7] [1, 1, 1, 1, 2]
    // CHECK: [[SLICE_U3:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 3] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_D3:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 10] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_U4:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 4] [1, 1, 1, 1, 1]
    // CHECK: [[SLICE_D4:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 12] [1, 1, 1, 1, 3]
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_U0]], [[SLICE_D0]], [[SLICE_U1]], [[SLICE_D1]], [[SLICE_U2]], [[SLICE_D2]], [[SLICE_U3]], [[SLICE_D3]], [[SLICE_U4]], [[SLICE_D4]])
    // CHECK-SAME: -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]]
}

// -----

// The indices shape could not meet Integer stride condition, so it will remain IE.ScatterNDUpdate.
// Non-integer stride for StridedConcat, but select pattern handles it: positions [0,3,6,9] from dim4 (size 15).
// CHECK-LABEL: @ConvertNonIntegerStrideScatterNDUpdateViaSelect
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1x15xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x1x5xf16>
func.func @ConvertNonIntegerStrideScatterNDUpdateViaSelect(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x4x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9]]]]]]> : tensor<1x1x1x1x4x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x4x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[CONCAT:%.+]] = IE.Concat(
    // CHECK-SAME: -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithSameSize
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1x5xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x1x5xf16>
func.func @ConvertToSliceConcatElementsUpdateWithSameSize(%arg0:  tensor<1x1x1x1x5xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x5xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]]]]]]>  : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x5xf16>

    return %0 : tensor<1x1x1x1x5xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return [[ARG_1]] : tensor<1x1x1x1x5xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithOneElement
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x1x1x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1x1xf16>
func.func @ConvertToSliceConcatElementsUpdateWithOneElement(%arg0:  tensor<1x1x1x1xf16>, %arg1 : tensor<1x1x1x1xf16> ) -> tensor<1x1x1x1xf16>{
    %cst = const.Declare tensor<1x1x1x1x4xsi32> = dense<[[[[[0,0,0,0]]]]]>  : tensor<1x1x1x1x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1xf16>, tensor<1x1x1x1x4xsi32>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    return %0 : tensor<1x1x1x1xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return [[ARG_1]] : tensor<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithReplaceOneElement
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x10x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x1x1xf16>
func.func @ConvertToSliceConcatElementsUpdateWithReplaceOneElement(%arg0:  tensor<1x10x1xf16>, %arg1 : tensor<1x1x1xf16> ) -> tensor<1x10x1xf16>{
    %cst = const.Declare tensor<1x1x1x3xsi32> = dense<0> : tensor<1x1x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x10x1xf16>, tensor<1x1x1x3xsi32>, tensor<1x1x1xf16> -> tensor<1x10x1xf16>

    return %0 : tensor<1x10x1xf16>

    // CHECK: [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 1, 0] [1, 9, 1] : tensor<1x10x1xf16> to tensor<1x9x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_1]], [[SLICE]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 1, 0]]} : tensor<1x1x1xf16>, tensor<1x9x1xf16> -> tensor<1x10x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x10x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithTwoSlice
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x326x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x7x1xf16>
func.func @ConvertToSliceConcatElementsUpdateWithTwoSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 249, 0]], [[0, 250, 0]], [[0, 251, 0]], [[0, 252, 0]], [[0, 253, 0]], [[0, 254, 0]], [[0, 255, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0] [1, 249, 1] : tensor<1x326x1xf16> to tensor<1x249x1xf16>
    // CHECK: [[SLICE_RIGHT:%.+]] = IE.Slice [[ARG_0]] [0, 256, 0] [1, 70, 1] : tensor<1x326x1xf16> to tensor<1x70x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_LEFT]], [[ARG_1]], [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 249, 0], [0, 256, 0]]} : tensor<1x249x1xf16>, tensor<1x7x1xf16>, tensor<1x70x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithRightSlice
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x326x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x7x1xf16>
func.func @ConvertToSliceConcatElementsUpdateWithRightSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 0, 0]], [[0, 1, 0]], [[0, 2, 0]], [[0, 3, 0]], [[0, 4, 0]], [[0, 5, 0]], [[0, 6, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_RIGHT:%.+]] = IE.Slice [[ARG_0]] [0, 7, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[ARG_1]], [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 7, 0]]} : tensor<1x7x1xf16>, tensor<1x319x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithLeftSlice
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x326x1xf16>, [[ARG_1:%[^:]+]]: tensor<1x7x1xf16>
func.func @ConvertToSliceConcatElementsUpdateWithLeftSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 319, 0]], [[0, 320, 0]], [[0, 321, 0]], [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_LEFT]], [[ARG_1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 319, 0]]} : tensor<1x319x1xf16>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithSplit
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x326x1xf16>, [[UPDATE:%.+]]: tensor<1x7x1xf16>)
func.func @ConvertToSliceConcatElementsUpdateWithSplit(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[
                [[0, 319, 0]], [[0, 320, 0]],
                [[0, 100, 0]],
                [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]
            ]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[UPDATE]] [0, 0, 0] [1, 2, 1] : tensor<1x7x1xf16> to tensor<1x2x1xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 321, 0] [1, 5, 1] : tensor<1x326x1xf16> to tensor<1x5x1xf16>
    // CHECK:   [[CONCAT_0:%.+]] = IE.Concat([[SLICE_1]], [[SLICE_0]], [[SLICE_2]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 319, 0], [0, 321, 0]]} : tensor<1x319x1xf16>, tensor<1x2x1xf16>, tensor<1x5x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[UPDATE]] [0, 2, 0] [1, 1, 1] : tensor<1x7x1xf16> to tensor<1x1x1xf16>
    // CHECK:   [[SLICE_4:%.+]] = IE.Slice [[CONCAT_0]] [0, 0, 0] [1, 100, 1] : tensor<1x326x1xf16> to tensor<1x100x1xf16>
    // CHECK:   [[SLICE_5:%.+]] = IE.Slice [[CONCAT_0]] [0, 101, 0] [1, 225, 1] : tensor<1x326x1xf16> to tensor<1x225x1xf16>
    // CHECK:   [[CONCAT_1:%.+]] = IE.Concat([[SLICE_4]], [[SLICE_3]], [[SLICE_5]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 100, 0], [0, 101, 0]]} : tensor<1x100x1xf16>, tensor<1x1x1xf16>, tensor<1x225x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   [[SLICE_6:%.+]] = IE.Slice [[UPDATE]] [0, 3, 0] [1, 4, 1] : tensor<1x7x1xf16> to tensor<1x4x1xf16>
    // CHECK:   [[SLICE_7:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 0] [1, 322, 1] : tensor<1x326x1xf16> to tensor<1x322x1xf16>
    // CHECK:   [[CONCAT_2:%.+]] = IE.Concat([[SLICE_7]], [[SLICE_6]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 322, 0]]} : tensor<1x322x1xf16>, tensor<1x4x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   return [[CONCAT_2]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatTensorUpdate
// CHECK-SAME:     [[ARG_0:%[^:]+]]: tensor<4x4x4xf16>
// CHECK-SAME:     [[ARG_1:%[^:]+]]: tensor<2x4x4xf16>
func.func @ConvertToSliceConcatTensorUpdate(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[1], [2]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK: [[SLICE_LEFT:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[SLICE_RIGHT:%.+]] = IE.Slice [[ARG_0]] [3, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_LEFT]], [[ARG_1]], [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [1, 0, 0], [3, 0, 0]]} : tensor<1x4x4xf16>, tensor<2x4x4xf16>, tensor<1x4x4xf16> -> tensor<4x4x4xf16>
    // CHECK: return [[CONCAT]] : tensor<4x4x4xf16>
}

// -----

// Non-sequential indices [2,1] — select pattern handles them by sorting positions.
// CHECK-LABEL: @ConvertNonSequentialTensorUpdateViaSelect
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<4x4x4xf16>, [[ARG_1:%[^:]+]]: tensor<2x4x4xf16>
func.func @ConvertNonSequentialTensorUpdateViaSelect(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[2], [1]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[CONCAT:%.+]] = IE.Concat(
    // CHECK-SAME: -> tensor<4x4x4xf16>
    // CHECK: return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ConvertToUpSamplingStridedConcatOnHW
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x1x4x4xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x1x2x2xf16>)
func.func @ConvertToUpSamplingStridedConcatOnHW(%arg0:  tensor<1x1x4x4xf16>, %arg1 : tensor<1x1x2x2xf16>) -> tensor<1x1x4x4xf16> {
    %indices = const.Declare tensor<1x1x2x2x4xsi32> = dense<[[[[[0, 0, 0, 0], [0, 0, 0, 2]], [[0, 0, 2, 0], [0, 0, 2, 2]]]]]> : tensor<1x1x2x2x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x1x4x4xf16>, tensor<1x1x2x2x4xsi32>, tensor<1x1x2x2xf16> -> tensor<1x1x4x4xf16>
    return %0 : tensor<1x1x4x4xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[UPSAMPLE:%.+]] = IE.Upsampling([[INPUT_DATA_1]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 1]>, upsampling_factor = [2, 1, 1]}
    // CHECK-SAME:    tensor<1x1x2x2xf16> -> tensor<1x1x2x4xf16>
    // CHECK: [[SLICE_INPUT_H:%.+]] = IE.StridedSlice([[INPUT_DATA_0]]) {begin_mask = [0, 0, 0, 0],
    // CHECK-SAME:    begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 1, 4, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>
    // CHECK-SAME:    shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]} : tensor<1x1x4x4xf16> -> tensor<1x1x2x4xf16>
    // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[UPSAMPLE]], [[SLICE_INPUT_H]])
    // CHECK-SAME:    {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x1x2x4xf16>, tensor<1x1x2x4xf16> -> tensor<1x1x4x4xf16>
    // CHECK: [[SLICE_UPDATE:%.+]] = IE.StridedSlice([[CONCAT_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:     ends_attr = [1, 1, 4, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]}
    // CHECK-SAME:    tensor<1x1x4x4xf16> -> tensor<1x1x4x2xf16>
    // CHECK: [[SLICE_INPUT_W:%.+]] = IE.StridedSlice([[INPUT_DATA_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 1, 4, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]}
    // CHECK-SAME:    tensor<1x1x4x4xf16> -> tensor<1x1x4x2xf16>
    // CHECK: [[CONCAT_RESULT:%.+]] = IE.Concat([[SLICE_UPDATE]], [[SLICE_INPUT_W]])
    // CHECK-SAME:     {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>}
    // CHECK-SAME:     tensor<1x1x4x2xf16>, tensor<1x1x4x2xf16> -> tensor<1x1x4x4xf16>
    // CHECK: return [[CONCAT_RESULT]] : tensor<1x1x4x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToUpSamplingStridedConcatOnCW
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x4x1x4xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x2x1x2xf16>)
func.func @ConvertToUpSamplingStridedConcatOnCW(%arg0:  tensor<1x4x1x4xf16>, %arg1 : tensor<1x2x1x2xf16>) -> tensor<1x4x1x4xf16> {
    %indices = const.Declare tensor<1x2x1x2x4xsi32> = dense<[[[[[0, 0, 0, 0], [0, 0, 0, 2]]], [[[0, 2, 0, 0], [0, 2, 0, 2]]]]]> : tensor<1x2x1x2x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x4x1x4xf16>, tensor<1x2x1x2x4xsi32>, tensor<1x2x1x2xf16> -> tensor<1x4x1x4xf16>
    return %0 : tensor<1x4x1x4xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[UPSAMPLE:%.+]] = IE.Upsampling([[INPUT_DATA_1]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 0], pads_width = [0, 1]>, upsampling_factor = [2, 1, 1]}
    // CHECK-SAME:    tensor<1x2x1x2xf16> -> tensor<1x2x1x4xf16>
    // CHECK: [[SLICE_INPUT_H:%.+]] = IE.StridedSlice([[INPUT_DATA_0]]) {begin_mask = [0, 0, 0, 0],
    // CHECK-SAME:    begins_attr = [0, 1, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 4, 1, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>
    // CHECK-SAME:    shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 1, 1]} : tensor<1x4x1x4xf16> -> tensor<1x2x1x4xf16>
    // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[UPSAMPLE]], [[SLICE_INPUT_H]])
    // CHECK-SAME:    {per_axis = #IE.Concat<axis = 1 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x1x4xf16>, tensor<1x2x1x4xf16> -> tensor<1x4x1x4xf16>
    // CHECK: [[SLICE_UPDATE:%.+]] = IE.StridedSlice([[CONCAT_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:     ends_attr = [1, 4, 1, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]}
    // CHECK-SAME:    tensor<1x4x1x4xf16> -> tensor<1x4x1x2xf16>
    // CHECK: [[SLICE_INPUT_W:%.+]] = IE.StridedSlice([[INPUT_DATA_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 4, 1, 4], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]}
    // CHECK-SAME:    tensor<1x4x1x4xf16> -> tensor<1x4x1x2xf16>
    // CHECK: [[CONCAT_RESULT:%.+]] = IE.Concat([[SLICE_UPDATE]], [[SLICE_INPUT_W]])
    // CHECK-SAME:     {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>}
    // CHECK-SAME:     tensor<1x4x1x2xf16>, tensor<1x4x1x2xf16> -> tensor<1x4x1x4xf16>
    // CHECK: return [[CONCAT_RESULT]] : tensor<1x4x1x4xf16>
}

// -----

// CHECK-LABEL: @ConvertToUpSamplingStridedConcatOnCH
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x4x4x1xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x2x2x1xf16>)
func.func @ConvertToUpSamplingStridedConcatOnCH(%arg0:  tensor<1x4x4x1xf16>, %arg1 : tensor<1x2x2x1xf16>) -> tensor<1x4x4x1xf16> {
    %indices = const.Declare tensor<1x2x2x1x4xsi32> = dense<[[[[[0, 0, 0, 0]], [[0, 0, 2, 0]]], [[[0, 2, 0, 0]], [[0, 2, 2, 0]]]]]> : tensor<1x2x2x1x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x4x4x1xf16>, tensor<1x2x2x1x4xsi32>, tensor<1x2x2x1xf16> -> tensor<1x4x4x1xf16>
    return %0 : tensor<1x4x4x1xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[UPSAMPLE:%.+]] = IE.Upsampling([[INPUT_DATA_1]]) {pad = #IE.UpsamplingPad<pads_channel = [0, 0], pads_height = [0, 1], pads_width = [0, 0]>, upsampling_factor = [1, 2, 1]}
    // CHECK-SAME:    tensor<1x2x2x1xf16> -> tensor<1x2x4x1xf16>
    // CHECK: [[SLICE_INPUT_H:%.+]] = IE.StridedSlice([[INPUT_DATA_0]]) {begin_mask = [0, 0, 0, 0],
    // CHECK-SAME:    begins_attr = [0, 1, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 4, 4, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>
    // CHECK-SAME:    shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 2, 1, 1]} : tensor<1x4x4x1xf16> -> tensor<1x2x4x1xf16>
    // CHECK: [[CONCAT_0:%.+]] = IE.Concat([[UPSAMPLE]], [[SLICE_INPUT_H]])
    // CHECK-SAME:    {per_axis = #IE.Concat<axis = 1 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x2x4x1xf16>, tensor<1x2x4x1xf16> -> tensor<1x4x4x1xf16>
    // CHECK: [[SLICE_UPDATE:%.+]] = IE.StridedSlice([[CONCAT_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:     ends_attr = [1, 4, 4, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]}
    // CHECK-SAME:    tensor<1x4x4x1xf16> -> tensor<1x4x2x1xf16>
    // CHECK: [[SLICE_INPUT_W:%.+]] = IE.StridedSlice([[INPUT_DATA_0]])
    // CHECK-SAME:    {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 1, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0]
    // CHECK-SAME:    ends_attr = [1, 4, 4, 1], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 2, 1]}
    // CHECK-SAME:    tensor<1x4x4x1xf16> -> tensor<1x4x2x1xf16>
    // CHECK: [[CONCAT_RESULT:%.+]] = IE.Concat([[SLICE_UPDATE]], [[SLICE_INPUT_W]])
    // CHECK-SAME:     {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>}
    // CHECK-SAME:     tensor<1x4x2x1xf16>, tensor<1x4x2x1xf16> -> tensor<1x4x4x1xf16>
    // CHECK: return [[CONCAT_RESULT]] : tensor<1x4x4x1xf16>
}

// -----

// CHECK-LABEL: @NotConvertToUpSamplingStridedConcatDueToNotIdentityMap
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x1x4x4xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x1x2x2xf16>)
func.func @NotConvertToUpSamplingStridedConcatDueToNotIdentityMap(%arg0:  tensor<1x1x4x4xf16>, %arg1 : tensor<1x1x2x2xf16>) -> tensor<1x1x4x4xf16> {
    %indices = const.Declare tensor<1x1x2x2x4xsi32> = dense<[[[[[0, 0, 0, 0], [0, 0, 2, 0]], [[0, 0, 0, 2], [0, 0, 2, 2]]]]]> : tensor<1x1x2x2x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x1x4x4xf16>, tensor<1x1x2x2x4xsi32>, tensor<1x1x2x2xf16> -> tensor<1x1x4x4xf16>
    return %0 : tensor<1x1x4x4xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x2x2x4xsi32>
    // CHECK: [[SCATTER:%.+]] = IE.ScatterNDUpdate([[INPUT_DATA_0]], [[INDICES]], [[INPUT_DATA_1]])
    // CHECK-SAME:              tensor<1x1x4x4xf16>, tensor<1x1x2x2x4xsi32>, tensor<1x1x2x2xf16> -> tensor<1x1x4x4xf16>
    // CHECK: return [[SCATTER]] : tensor<1x1x4x4xf16>
}

// -----

// CHECK-LABEL: @NotConvertToUpSamplingStridedConcatDueToScaleFactorNotInteger
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x1x4x5xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x1x2x2xf16>)
func.func @NotConvertToUpSamplingStridedConcatDueToScaleFactorNotInteger(%arg0:  tensor<1x1x4x5xf16>, %arg1 : tensor<1x1x2x2xf16>) -> tensor<1x1x4x5xf16> {
    %indices = const.Declare tensor<1x1x2x2x4xsi32> = dense<[[[[[0, 0, 0, 0], [0, 0, 0, 2]], [[0, 0, 2, 0], [0, 0, 2, 2]]]]]> : tensor<1x1x2x2x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x1x4x5xf16>, tensor<1x1x2x2x4xsi32>, tensor<1x1x2x2xf16> -> tensor<1x1x4x5xf16>
    return %0 : tensor<1x1x4x5xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x2x2x4xsi32>
    // CHECK: [[SCATTER:%.+]] = IE.ScatterNDUpdate([[INPUT_DATA_0]], [[INDICES]], [[INPUT_DATA_1]])
    // CHECK-SAME:              tensor<1x1x4x5xf16>, tensor<1x1x2x2x4xsi32>, tensor<1x1x2x2xf16> -> tensor<1x1x4x5xf16>
    // CHECK: return [[SCATTER]] : tensor<1x1x4x5xf16>
}

// -----

// CHECK-LABEL: @NotConvertToUpSamplingStridedConcatDueToNotEltwiseCase
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x4x4x1xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x2x2x1xf16>)
func.func @NotConvertToUpSamplingStridedConcatDueToNotEltwiseCase(%arg0:  tensor<1x4x4x1xf16>, %arg1 : tensor<1x2x2x1xf16>) -> tensor<1x4x4x1xf16> {
    %indices = const.Declare tensor<1x2x2x1x3xsi32> = dense<[[[[[0, 0, 0]], [[0, 2, 0]]], [[[2, 0, 0]], [[2, 2, 0]]]]]> : tensor<1x2x2x1x3xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x4x4x1xf16>, tensor<1x2x2x1x3xsi32>, tensor<1x2x2x1xf16> -> tensor<1x4x4x1xf16>
    return %0 : tensor<1x4x4x1xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x2x2x1x3xsi32>
    // CHECK: [[SCATTER:%.+]] = IE.ScatterNDUpdate([[INPUT_DATA_0]], [[INDICES]], [[INPUT_DATA_1]])
    // CHECK-SAME:              tensor<1x4x4x1xf16>, tensor<1x2x2x1x3xsi32>, tensor<1x2x2x1xf16> -> tensor<1x4x4x1xf16>
    // CHECK: return [[SCATTER]] : tensor<1x4x4x1xf16>
}

// -----

// CHECK-LABEL: @NotConvertToUpSamplingStridedConcatDueToInputNot4D
// CHECK-SAME:  ([[INPUT_DATA_0:%.+]]: tensor<1x4x4xf16>, [[INPUT_DATA_1:%.+]]: tensor<1x2x2xf16>)
func.func @NotConvertToUpSamplingStridedConcatDueToInputNot4D(%arg0:  tensor<1x4x4xf16>, %arg1 : tensor<1x2x2xf16>) -> tensor<1x4x4xf16> {
    %indices = const.Declare tensor<1x2x2x3xsi32> = dense<[[[[0, 0, 0], [0, 0, 2]], [[0, 2, 0], [0, 2, 2]]]]> : tensor<1x2x2x3xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %indices, %arg1) : tensor<1x4x4xf16>, tensor<1x2x2x3xsi32>, tensor<1x2x2xf16> -> tensor<1x4x4xf16>
    return %0 : tensor<1x4x4xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x2x2x3xsi32>
    // CHECK: [[SCATTER:%.+]] = IE.ScatterNDUpdate([[INPUT_DATA_0]], [[INDICES]], [[INPUT_DATA_1]])
    // CHECK-SAME:              tensor<1x4x4xf16>, tensor<1x2x2x3xsi32>, tensor<1x2x2xf16> -> tensor<1x4x4xf16>
    // CHECK: return [[SCATTER]] : tensor<1x4x4xf16>
}

// -----

// CHECK-LABEL: @Convert4DUpdateDataToSliceConcat
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x5x6xf16>, [[UPDATE:%.+]]: tensor<1x2x2x2xf16>)
func.func @Convert4DUpdateDataToSliceConcat(%arg0:  tensor<1x4x5x6xf16>, %arg1: tensor<1x2x2x2xf16> ) -> tensor<1x4x5x6xf16>{
    %cst = const.Declare tensor<1x2x2x2x4xsi32> = dense<[[[[[0, 1, 1, 1], [0, 1, 1, 2]],
                                                          [[0, 1, 2, 1], [0, 1, 2, 2]]],
                                                         [[[0, 2, 1, 1], [0, 2, 1, 2]],
                                                          [[0, 2, 2, 1], [0, 2, 2, 2]]]]]> : tensor<1x2x2x2x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x5x6xf16>, tensor<1x2x2x2x4xsi32>, tensor<1x2x2x2xf16> -> tensor<1x4x5x6xf16>
    return %0 : tensor<1x4x5x6xf16>

    // CHECK-NOT:   IE.ScatterNDUpdate
    // CHECK:   [[SLICE_W_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 1, 1, 0] [1, 2, 2, 1] : tensor<1x4x5x6xf16> to tensor<1x2x2x1xf16>
    // CHECK:   [[SLICE_W_END:%.+]] = IE.Slice [[INPUT]] [0, 1, 1, 3] [1, 2, 2, 3] : tensor<1x4x5x6xf16> to tensor<1x2x2x3xf16>
    // CHECK:   [[CONCAT_W:%.+]] = IE.Concat([[SLICE_W_BEGIN]], [[UPDATE]], [[SLICE_W_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 3]]} : tensor<1x2x2x1xf16>, tensor<1x2x2x2xf16>, tensor<1x2x2x3xf16> -> tensor<1x2x2x6xf16>

    // CHECK:   [[SLICE_H_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 2, 1, 6] : tensor<1x4x5x6xf16> to tensor<1x2x1x6xf16>
    // CHECK:   [[SLICE_H_END:%.+]] = IE.Slice [[INPUT]] [0, 1, 3, 0] [1, 2, 2, 6] : tensor<1x4x5x6xf16> to tensor<1x2x2x6xf16>
    // CHECK:   [[CONCAT_H:%.+]] = IE.Concat([[SLICE_H_BEGIN]], [[CONCAT_W]], [[SLICE_H_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 3, 0]]} : tensor<1x2x1x6xf16>, tensor<1x2x2x6xf16>, tensor<1x2x2x6xf16> -> tensor<1x2x5x6xf16>

    // CHECK:   [[SLICE_C_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 5, 6] : tensor<1x4x5x6xf16> to tensor<1x1x5x6xf16>
    // CHECK:   [[SLICE_C_END:%.+]] = IE.Slice [[INPUT]] [0, 3, 0, 0] [1, 1, 5, 6] : tensor<1x4x5x6xf16> to tensor<1x1x5x6xf16>
    // CHECK:   [[CONCAT_C:%.+]] = IE.Concat([[SLICE_C_BEGIN]], [[CONCAT_H]], [[SLICE_C_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 3, 0, 0]]} : tensor<1x1x5x6xf16>, tensor<1x2x5x6xf16>, tensor<1x1x5x6xf16> -> tensor<1x4x5x6xf16>

    // CHECK:   return [[CONCAT_C]] : tensor<1x4x5x6xf16>
}

// -----

// CHECK-LABEL: @Convert5DUpdateDataToSliceConcat
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x5x6x7xf16>, [[UPDATE:%.+]]: tensor<1x2x1x2x2xf16>)
func.func @Convert5DUpdateDataToSliceConcat(%arg0:  tensor<1x4x5x6x7xf16>, %arg1: tensor<1x2x1x2x2xf16> ) -> tensor<1x4x5x6x7xf16>{
    %cst = const.Declare tensor<1x2x1x2x2x5xsi32> = dense<[[[[[[0, 2, 3, 0, 2], [0, 2, 3, 0, 3]],
                                                              [[0, 2, 3, 1, 2], [0, 2, 3, 1, 3]]]],
                                                            [[[[0, 3, 3, 0, 2], [0, 3, 3, 0, 3]],
                                                              [[0, 3, 3, 1, 2], [0, 3, 3, 1, 3]]]]]]> : tensor<1x2x1x2x2x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x5x6x7xf16>, tensor<1x2x1x2x2x5xsi32>, tensor<1x2x1x2x2xf16> -> tensor<1x4x5x6x7xf16>
    return %0 : tensor<1x4x5x6x7xf16>

    // CHECK-NOT:   IE.ScatterNDUpdate
    // CHECK:   [[SLICE_W_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 0, 0] [1, 2, 1, 2, 2] : tensor<1x4x5x6x7xf16> to tensor<1x2x1x2x2xf16>
    // CHECK:   [[SLICE_W_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 0, 4] [1, 2, 1, 2, 3] : tensor<1x4x5x6x7xf16> to tensor<1x2x1x2x3xf16>
    // CHECK:   [[CONCAT_W:%.+]] = IE.Concat([[SLICE_W_BEGIN]], [[UPDATE]], [[SLICE_W_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4]]} : tensor<1x2x1x2x2xf16>, tensor<1x2x1x2x2xf16>, tensor<1x2x1x2x3xf16> -> tensor<1x2x1x2x7xf16>

    // CHECK:   [[SLICE_H_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 2, 0] [1, 2, 1, 4, 7] : tensor<1x4x5x6x7xf16> to tensor<1x2x1x4x7xf16>
    // CHECK:   [[CONCAT_H:%.+]] = IE.Concat([[CONCAT_W]], [[SLICE_H_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0], [0, 0, 0, 2, 0]]} : tensor<1x2x1x2x7xf16>, tensor<1x2x1x4x7xf16> -> tensor<1x2x1x6x7xf16>

    // CHECK:   [[SLICE_D_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 2, 0, 0, 0] [1, 2, 3, 6, 7] : tensor<1x4x5x6x7xf16> to tensor<1x2x3x6x7xf16>
    // CHECK:   [[SLICE_D_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 4, 0, 0] [1, 2, 1, 6, 7] : tensor<1x4x5x6x7xf16> to tensor<1x2x1x6x7xf16>
    // CHECK:   [[CONCAT_D:%.+]] = IE.Concat([[SLICE_D_BEGIN]], [[CONCAT_H]], [[SLICE_D_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0], [0, 0, 3, 0, 0], [0, 0, 4, 0, 0]]} : tensor<1x2x3x6x7xf16>, tensor<1x2x1x6x7xf16>, tensor<1x2x1x6x7xf16> -> tensor<1x2x5x6x7xf16>

    // CHECK:   [[SLICE_C_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 2, 5, 6, 7] : tensor<1x4x5x6x7xf16> to tensor<1x2x5x6x7xf16>
    // CHECK:   [[CONCAT_C:%.+]] = IE.Concat([[SLICE_C_BEGIN]], [[CONCAT_D]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0]]} : tensor<1x2x5x6x7xf16>, tensor<1x2x5x6x7xf16> -> tensor<1x4x5x6x7xf16>

    // CHECK:   return [[CONCAT_C]] : tensor<1x4x5x6x7xf16>
}

// -----

// CHECK-LABEL: @Convert5DUpdateDataToSliceConcatSmallIndices
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x5x6x7x3xf16>, [[UPDATE:%.+]]: tensor<1x2x1x2x2x3xf16>)
func.func @Convert5DUpdateDataToSliceConcatSmallIndices(%arg0:  tensor<1x4x5x6x7x3xf16>, %arg1: tensor<1x2x1x2x2x3xf16> ) -> tensor<1x4x5x6x7x3xf16>{
    %cst = const.Declare tensor<1x2x1x2x2x5xsi32> = dense<[[[[[[0, 2, 3, 0, 2], [0, 2, 3, 0, 3]],
                                                              [[0, 2, 3, 1, 2], [0, 2, 3, 1, 3]]]],
                                                            [[[[0, 3, 3, 0, 2], [0, 3, 3, 0, 3]],
                                                              [[0, 3, 3, 1, 2], [0, 3, 3, 1, 3]]]]]]> : tensor<1x2x1x2x2x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x5x6x7x3xf16>, tensor<1x2x1x2x2x5xsi32>, tensor<1x2x1x2x2x3xf16> -> tensor<1x4x5x6x7x3xf16>
    return %0 : tensor<1x4x5x6x7x3xf16>

    // CHECK-NOT:   IE.ScatterNDUpdate
    // CHECK:   [[SLICE_W_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 0, 0, 0] [1, 2, 1, 2, 2, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x1x2x2x3xf16>
    // CHECK:   [[SLICE_W_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 0, 4, 0] [1, 2, 1, 2, 3, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x1x2x3x3xf16>
    // CHECK:   [[CONCAT_W:%.+]] = IE.Concat([[SLICE_W_BEGIN]], [[UPDATE]], [[SLICE_W_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 4, 0]]} : tensor<1x2x1x2x2x3xf16>, tensor<1x2x1x2x2x3xf16>, tensor<1x2x1x2x3x3xf16> -> tensor<1x2x1x2x7x3xf16>

    // CHECK:   [[SLICE_H_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 3, 2, 0, 0] [1, 2, 1, 4, 7, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x1x4x7x3xf16>
    // CHECK:   [[CONCAT_H:%.+]] = IE.Concat([[CONCAT_W]], [[SLICE_H_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0]]} : tensor<1x2x1x2x7x3xf16>, tensor<1x2x1x4x7x3xf16> -> tensor<1x2x1x6x7x3xf16>

    // CHECK:   [[SLICE_D_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 2, 0, 0, 0, 0] [1, 2, 3, 6, 7, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x3x6x7x3xf16>
    // CHECK:   [[SLICE_D_END:%.+]] = IE.Slice [[INPUT]] [0, 2, 4, 0, 0, 0] [1, 2, 1, 6, 7, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x1x6x7x3xf16>
    // CHECK:   [[CONCAT_D:%.+]] = IE.Concat([[SLICE_D_BEGIN]], [[CONCAT_H]], [[SLICE_D_END]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0, 0], [0, 0, 3, 0, 0, 0], [0, 0, 4, 0, 0, 0]]} : tensor<1x2x3x6x7x3xf16>, tensor<1x2x1x6x7x3xf16>, tensor<1x2x1x6x7x3xf16> -> tensor<1x2x5x6x7x3xf16>

    // CHECK:   [[SLICE_C_BEGIN:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0, 0] [1, 2, 5, 6, 7, 3] : tensor<1x4x5x6x7x3xf16> to tensor<1x2x5x6x7x3xf16>
    // CHECK:   [[CONCAT_C:%.+]] = IE.Concat([[SLICE_C_BEGIN]], [[CONCAT_D]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0]]} : tensor<1x2x5x6x7x3xf16>, tensor<1x2x5x6x7x3xf16> -> tensor<1x4x5x6x7x3xf16>

    // CHECK:   return [[CONCAT_C]] : tensor<1x4x5x6x7x3xf16>
}

// -----

// CHECK-LABEL: @SplitToMultiScatterNDUpdateOpWithTensorReplace
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x326x1xf16>, [[UPDATE:%.+]]: tensor<1x7x1xf16>)
func.func @SplitToMultiScatterNDUpdateOpWithTensorReplace(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[
                    [[0, 249, 0]], [[0, 250, 0]], [[0, 251, 0]],
                    [[0, 258, 0]], [[0, 259, 0]], [[0, 260, 0]],
                    [[0, 300, 0]]
                ]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK:   [[SLICE_0:%.+]] = IE.Slice [[UPDATE]] [0, 0, 0] [1, 3, 1] : tensor<1x7x1xf16> to tensor<1x3x1xf16>
    // CHECK:   [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [1, 249, 1] : tensor<1x326x1xf16> to tensor<1x249x1xf16>
    // CHECK:   [[SLICE_2:%.+]] = IE.Slice [[INPUT]] [0, 252, 0] [1, 74, 1] : tensor<1x326x1xf16> to tensor<1x74x1xf16>
    // CHECK:   [[CONCAT_0:%.+]] = IE.Concat([[SLICE_1]], [[SLICE_0]], [[SLICE_2]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 249, 0], [0, 252, 0]]} : tensor<1x249x1xf16>, tensor<1x3x1xf16>, tensor<1x74x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   [[SLICE_3:%.+]] = IE.Slice [[UPDATE]] [0, 3, 0] [1, 3, 1] : tensor<1x7x1xf16> to tensor<1x3x1xf16>
    // CHECK:   [[SLICE_4:%.+]] = IE.Slice [[CONCAT_0]] [0, 0, 0] [1, 258, 1] : tensor<1x326x1xf16> to tensor<1x258x1xf16>
    // CHECK:   [[SLICE_5:%.+]] = IE.Slice [[CONCAT_0]] [0, 261, 0] [1, 65, 1] : tensor<1x326x1xf16> to tensor<1x65x1xf16>
    // CHECK:   [[CONCAT_1:%.+]] = IE.Concat([[SLICE_4]], [[SLICE_3]], [[SLICE_5]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 258, 0], [0, 261, 0]]} : tensor<1x258x1xf16>, tensor<1x3x1xf16>, tensor<1x65x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   [[SLICE_6:%.+]] = IE.Slice [[UPDATE]] [0, 6, 0] [1, 1, 1] : tensor<1x7x1xf16> to tensor<1x1x1xf16>
    // CHECK:   [[SLICE_7:%.+]] = IE.Slice [[CONCAT_1]] [0, 0, 0] [1, 300, 1] : tensor<1x326x1xf16> to tensor<1x300x1xf16>
    // CHECK:   [[SLICE_8:%.+]] = IE.Slice [[CONCAT_1]] [0, 301, 0] [1, 25, 1] : tensor<1x326x1xf16> to tensor<1x25x1xf16>
    // CHECK:   [[CONCAT_2:%.+]] = IE.Concat([[SLICE_7]], [[SLICE_6]], [[SLICE_8]]) {
    // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0], [0, 300, 0], [0, 301, 0]]} : tensor<1x300x1xf16>, tensor<1x1x1xf16>, tensor<1x25x1xf16> -> tensor<1x326x1xf16>

    // CHECK:   return [[CONCAT_2]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @SplitToMultiScatterNDUpdateOpElementReplace
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<326xf16>, [[UPDATE:%.+]]: tensor<7xf16>)
func.func @SplitToMultiScatterNDUpdateOpElementReplace(%arg0:  tensor<326xf16>, %arg1 : tensor<7xf16> ) -> tensor<326xf16>{
    %cst = const.Declare tensor<7x1xsi32> = dense<[
                    [249], [250], [251],
                    [258], [259], [260],
                    [300]
                ]> : tensor<7x1xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<326xf16>, tensor<7x1xsi32>, tensor<7xf16> -> tensor<326xf16>

    return %0 : tensor<326xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK:   [[S0:%.+]] = IE.Slice [[INPUT]] [0] [249] : tensor<326xf16> to tensor<249xf16>
    // CHECK:   [[U0:%.+]] = IE.Slice [[UPDATE]] [0] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[U1:%.+]] = IE.Slice [[UPDATE]] [1] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[U2:%.+]] = IE.Slice [[UPDATE]] [2] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[S1:%.+]] = IE.Slice [[INPUT]] [252] [6] : tensor<326xf16> to tensor<6xf16>
    // CHECK:   [[U3:%.+]] = IE.Slice [[UPDATE]] [3] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[U4:%.+]] = IE.Slice [[UPDATE]] [4] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[U5:%.+]] = IE.Slice [[UPDATE]] [5] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[S2:%.+]] = IE.Slice [[INPUT]] [261] [39] : tensor<326xf16> to tensor<39xf16>
    // CHECK:   [[U6:%.+]] = IE.Slice [[UPDATE]] [6] [1] : tensor<7xf16> to tensor<1xf16>
    // CHECK:   [[S3:%.+]] = IE.Slice [[INPUT]] [301] [25] : tensor<326xf16> to tensor<25xf16>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[S0]], [[U0]], [[U1]], [[U2]], [[S1]], [[U3]], [[U4]], [[U5]], [[S2]], [[U6]], [[S3]])
    // CHECK-SAME: -> tensor<326xf16>
    // CHECK:   return [[CONCAT]] : tensor<326xf16>
}

// -----

// CHECK-LABEL: @SplitToMultiScatterNDUpdateOpWithIndicesRankSmallerThanInput
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x326x16xf16>, [[UPDATE:%.+]]: tensor<1x7x16xf16>)
func.func @SplitToMultiScatterNDUpdateOpWithIndicesRankSmallerThanInput(%arg0:  tensor<1x326x16xf16>, %arg1 : tensor<1x7x16xf16> ) -> tensor<1x326x16xf16>{
    %cst = const.Declare tensor<1x7x2xsi32> = dense<[[
                    [0, 249], [0, 250], [0, 251],
                    [0, 258], [0, 259], [0, 260],
                    [0, 300]
                ]]> : tensor<1x7x2xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x16xf16>, tensor<1x7x2xsi32>, tensor<1x7x16xf16> -> tensor<1x326x16xf16>

    return %0 : tensor<1x326x16xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK:   [[S0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [1, 249, 16] : tensor<1x326x16xf16> to tensor<1x249x16xf16>
    // CHECK:   [[U0:%.+]] = IE.Slice [[UPDATE]] [0, 0, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[U1:%.+]] = IE.Slice [[UPDATE]] [0, 1, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[U2:%.+]] = IE.Slice [[UPDATE]] [0, 2, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[S1:%.+]] = IE.Slice [[INPUT]] [0, 252, 0] [1, 6, 16] : tensor<1x326x16xf16> to tensor<1x6x16xf16>
    // CHECK:   [[U3:%.+]] = IE.Slice [[UPDATE]] [0, 3, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[U4:%.+]] = IE.Slice [[UPDATE]] [0, 4, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[U5:%.+]] = IE.Slice [[UPDATE]] [0, 5, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[S2:%.+]] = IE.Slice [[INPUT]] [0, 261, 0] [1, 39, 16] : tensor<1x326x16xf16> to tensor<1x39x16xf16>
    // CHECK:   [[U6:%.+]] = IE.Slice [[UPDATE]] [0, 6, 0] [1, 1, 16] : tensor<1x7x16xf16> to tensor<1x1x16xf16>
    // CHECK:   [[S3:%.+]] = IE.Slice [[INPUT]] [0, 301, 0] [1, 25, 16] : tensor<1x326x16xf16> to tensor<1x25x16xf16>
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[S0]], [[U0]], [[U1]], [[U2]], [[S1]], [[U3]], [[U4]], [[U5]], [[S2]], [[U6]], [[S3]])
    // CHECK-SAME: -> tensor<1x326x16xf16>
    // CHECK:   return [[CONCAT]] : tensor<1x326x16xf16>
}


// CHECK-LABEL: @ConvertScatterElementsUpdateSumToAdd
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5xf16>, [[UPDATE:%.+]]: tensor<1x5xf16>)
func.func @ConvertScatterElementsUpdateSumToAdd(%input: tensor<1x5xf16>, %update: tensor<1x5xf16>) -> tensor<1x5xf16> {
    %indices = const.Declare tensor<1x5xsi32> = dense<[[0,0,0,0,0]]> : tensor<1x5xsi32>
    %0 = IE.ScatterElementsUpdate(%input, %indices, %update) {axis_value = 0 : i64, reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true} : tensor<1x5xf16>, tensor<1x5xsi32>, tensor<1x5xf16> -> tensor<1x5xf16>
    return %0 : tensor<1x5xf16>

    // CHECK: [[ADD:%.+]] = IE.Add([[INPUT]], [[UPDATE]])
    // CHECK: return [[ADD]]
}


// CHECK-LABEL: @ConvertScatterElementsUpdateMulToMultiply
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5xf16>, [[UPDATE:%.+]]: tensor<1x5xf16>)
func.func @ConvertScatterElementsUpdateMulToMultiply(%input: tensor<1x5xf16>, %update: tensor<1x5xf16>) -> tensor<1x5xf16> {
    %indices = const.Declare tensor<1x5xsi32> = dense<[[0,0,0,0,0]]> : tensor<1x5xsi32>
    %0 = IE.ScatterElementsUpdate(%input, %indices, %update) {axis_value = 0 : i64, reduction = #IE.scatter_elements_update_reduction_type<PROD>, use_init_val = true} : tensor<1x5xf16>, tensor<1x5xsi32>, tensor<1x5xf16> -> tensor<1x5xf16>
    return %0 : tensor<1x5xf16>

    // CHECK: [[MUL:%.+]] = IE.Multiply([[INPUT]], [[UPDATE]])
    // CHECK: return [[MUL]]
}


// -----

// CHECK-LABEL: @DoNotConvertScatterElementsUpdateNonZeroIndices
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5xf16>, [[UPDATE:%.+]]: tensor<1x5xf16>)
func.func @DoNotConvertScatterElementsUpdateNonZeroIndices(%input: tensor<1x5xf16>, %update: tensor<1x5xf16>) -> tensor<1x5xf16> {
    %indices = const.Declare tensor<1x5xsi32> = dense<[[0,1,0,0,0]]> : tensor<1x5xsi32>
    %0 = IE.ScatterElementsUpdate(%input, %indices, %update) {axis_value = 0 : i64, reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true} : tensor<1x5xf16>, tensor<1x5xsi32>, tensor<1x5xf16> -> tensor<1x5xf16>
    return %0 : tensor<1x5xf16>

    // CHECK: [[SCATTER:%.+]] = IE.ScatterElementsUpdate
    // CHECK: return [[SCATTER]]
}

// -----

// CHECK-LABEL: @DoNotConvertScatterElementsUpdateAxisNonZero
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5xf16>, [[UPDATE:%.+]]: tensor<1x5xf16>)
func.func @DoNotConvertScatterElementsUpdateAxisNonZero(%input: tensor<1x5xf16>, %update: tensor<1x5xf16>) -> tensor<1x5xf16> {
    %indices = const.Declare tensor<1x5xsi32> = dense<[[0,0,0,0,0]]> : tensor<1x5xsi32>
    %0 = IE.ScatterElementsUpdate(%input, %indices, %update) {axis_value = 1 : i64, reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true} : tensor<1x5xf16>, tensor<1x5xsi32>, tensor<1x5xf16> -> tensor<1x5xf16>
    return %0 : tensor<1x5xf16>

    // CHECK: [[SCATTER:%.+]] = IE.ScatterElementsUpdate
    // CHECK: return [[SCATTER]]
}


// -----

// CHECK-LABEL: @DoNotConvertScatterElementsUpdateNonConstIndices
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x5xf16>, [[UPDATE:%.+]]: tensor<1x5xf16>, [[INDICES:%.+]]: tensor<1x5xsi32>)
func.func @DoNotConvertScatterElementsUpdateNonConstIndices(%input: tensor<1x5xf16>, %update: tensor<1x5xf16>, %indices: tensor<1x5xsi32>) -> tensor<1x5xf16> {
    %0 = IE.ScatterElementsUpdate(%input, %indices, %update) {axis_value = 0 : i64, reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true} : tensor<1x5xf16>, tensor<1x5xsi32>, tensor<1x5xf16> -> tensor<1x5xf16>
    return %0 : tensor<1x5xf16>

    // CHECK: [[SCATTER:%.+]] = IE.ScatterElementsUpdate
    // CHECK: return [[SCATTER]]
}

// -----

// CHECK-LABEL: @ConvertScatterNDUpdateNegativeIndicesNormalized
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x4x4xf16>, [[UPDATE:%.+]]: tensor<1x1x3xf16>)
func.func @ConvertScatterNDUpdateNegativeIndicesNormalized(%arg0: tensor<1x4x4xf16>, %arg1: tensor<1x1x3xf16>) -> tensor<1x4x4xf16> {
    %cst = const.Declare tensor<1x1x3x3xsi32> = dense<[[[[0, -1, 0], [0, -1, 1], [0, -1, 2]]]]> : tensor<1x1x3x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x4xf16>, tensor<1x1x3x3xsi32>, tensor<1x1x3xf16> -> tensor<1x4x4xf16>
    return %0 : tensor<1x4x4xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK-DAG: [[TAIL:%.+]] = IE.Slice [[INPUT]] [0, 3, 3] [1, 1, 1] : tensor<1x4x4xf16> to tensor<1x1x1xf16>
    // CHECK-DAG: [[ROW:%.+]] = IE.Concat([[UPDATE]], [[TAIL]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 3]]} : tensor<1x1x3xf16>, tensor<1x1x1xf16> -> tensor<1x1x4xf16>
    // CHECK-DAG: [[HEAD:%.+]] = IE.Slice [[INPUT]] [0, 0, 0] [1, 3, 4] : tensor<1x4x4xf16> to tensor<1x3x4xf16>
    // CHECK: [[RESULT:%.+]] = IE.Concat([[HEAD]], [[ROW]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 3, 0]]} : tensor<1x3x4xf16>, tensor<1x1x4xf16> -> tensor<1x4x4xf16>
    // CHECK: return [[RESULT]] : tensor<1x4x4xf16>
}

// -----

// CHECK-LABEL: @DoNotSplitScatterNDUpdateAxisMismatch
// CHECK-SAME: ([[INPUT:%.+]]: tensor<1x4x4xf16>, [[UPDATE:%.+]]: tensor<1x1x3xf16>)
func.func @DoNotSplitScatterNDUpdateAxisMismatch(%arg0: tensor<1x4x4xf16>, %arg1: tensor<1x1x3xf16>) -> tensor<1x4x4xf16> {
    %cst = const.Declare tensor<1x1x3x3xsi32> = dense<[[[[0, 0, 0], [0, 2, 0], [0, 3, 0]]]]> : tensor<1x1x3x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x4xf16>, tensor<1x1x3x3xsi32>, tensor<1x1x3xf16> -> tensor<1x4x4xf16>
    return %0 : tensor<1x4x4xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x3x3xsi32>
    // CHECK: [[RESULT:%.+]] = IE.ScatterNDUpdate([[INPUT]], [[CST]], [[UPDATE]])
    // CHECK: return [[RESULT]] : tensor<1x4x4xf16>
}

// -----

// 1D stride-3 arithmetic indices [1,4,7] on rank-1 input
// CHECK-LABEL: @ConvertScatterNDArithmeticStride3
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<15xf16>, [[ARG_1:%[^:]+]]: tensor<3xf16>
func.func @ConvertScatterNDArithmeticStride3(%arg0: tensor<15xf16>, %arg1: tensor<3xf16>) -> tensor<15xf16> {
    %cst = const.Declare tensor<3x1xsi64> = dense<[[1], [4], [7]]> : tensor<3x1xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<15xf16>, tensor<3x1xsi64>, tensor<3xf16> -> tensor<15xf16>
    return %0 : tensor<15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK-NOT: IE.StridedSlice
    // CHECK: [[HEAD:%.+]] = IE.Slice [[ARG_0]] [0] [1] : tensor<15xf16> to tensor<1xf16>
    // CHECK: [[BODY:%.+]] = IE.Slice [[ARG_0]] [1] [9] : tensor<15xf16> to tensor<9xf16>
    // CHECK: [[GROUPS:%.+]] = IE.AffineReshape([[BODY]]) {{.*}} : tensor<9xf16> -> tensor<3x3xf16>
    // CHECK: [[KEEP:%.+]] = IE.Slice [[GROUPS]] [0, 1] [3, 2] : tensor<3x3xf16> to tensor<3x2xf16>
    // CHECK: [[UPD:%.+]] = IE.AffineReshape([[ARG_1]]) {{.*}} : tensor<3xf16> -> tensor<3x1xf16>
    // CHECK: [[NEW:%.+]] = IE.Concat([[UPD]], [[KEEP]]) {{.*}} : tensor<3x1xf16>, tensor<3x2xf16> -> tensor<3x3xf16>
    // CHECK: [[BODY_NEW:%.+]] = IE.AffineReshape([[NEW]]) {{.*}} : tensor<3x3xf16> -> tensor<9xf16>
    // CHECK: [[TAIL:%.+]] = IE.Slice [[ARG_0]] [10] [5] : tensor<15xf16> to tensor<5xf16>
    // CHECK: [[OUT:%.+]] = IE.Concat([[HEAD]], [[BODY_NEW]], [[TAIL]]) {{.*}} : tensor<1xf16>, tensor<9xf16>, tensor<5xf16> -> tensor<15xf16>
    // CHECK: return [[OUT]] : tensor<15xf16>
}

// -----

// 2D strided grid (outerCount > 1): indices=[1,4,7,11,14,17] map to base=1, stride=3, M=3, N=2.
// CHECK-LABEL: @ConvertScatterNDArithmetic2DGrid
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<21xf16>, [[ARG_1:%[^:]+]]: tensor<6xf16>
func.func @ConvertScatterNDArithmetic2DGrid(%arg0: tensor<21xf16>, %arg1: tensor<6xf16>) -> tensor<21xf16> {
    %cst = const.Declare tensor<6x1xsi64> = dense<[[1], [4], [7], [11], [14], [17]]> : tensor<6x1xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<21xf16>, tensor<6x1xsi64>, tensor<6xf16> -> tensor<21xf16>
    return %0 : tensor<21xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK-NOT: IE.StridedSlice
    // CHECK: [[HEAD:%.+]] = IE.Slice [[ARG_0]] [0] [1] : tensor<21xf16> to tensor<1xf16>
    // CHECK: [[HEAD_BODY:%.+]] = IE.Slice [[ARG_0]] [1] [10] : tensor<21xf16> to tensor<10xf16>
    // CHECK: [[HEAD_UPD:%.+]] = IE.Slice [[ARG_1]] [0] [3] : tensor<6xf16> to tensor<3xf16>
    // CHECK: [[HEAD_BODY_2D:%.+]] = IE.AffineReshape([[HEAD_BODY]]) {{.*}} : tensor<10xf16> -> tensor<1x10xf16>
    // CHECK: [[STRIDE_REGION:%.+]] = IE.Slice [[HEAD_BODY_2D]] [0, 0] [1, 9] : tensor<1x10xf16> to tensor<1x9xf16>
    // CHECK: [[GROUPS:%.+]] = IE.Reshape([[STRIDE_REGION]]) {{.*}} : tensor<1x9xf16> -> tensor<3x3xf16>
    // CHECK: [[KEEP:%.+]] = IE.Slice [[GROUPS]] [0, 1] [3, 2] : tensor<3x3xf16> to tensor<3x2xf16>
    // CHECK: [[UPD:%.+]] = IE.AffineReshape([[HEAD_UPD]]) {{.*}} : tensor<3xf16> -> tensor<3x1xf16>
    // CHECK: [[NEW:%.+]] = IE.Concat([[UPD]], [[KEEP]]) {{.*}} : tensor<3x1xf16>, tensor<3x2xf16> -> tensor<3x3xf16>
    // CHECK: [[NEW_2D:%.+]] = IE.Reshape([[NEW]]) {{.*}} : tensor<3x3xf16> -> tensor<1x9xf16>
    // CHECK: [[GAP:%.+]] = IE.Slice [[HEAD_BODY_2D]] [0, 9] [1, 1] : tensor<1x10xf16> to tensor<1x1xf16>
    // CHECK: [[ROW:%.+]] = IE.Concat([[NEW_2D]], [[GAP]]) {{.*}} : tensor<1x9xf16>, tensor<1x1xf16> -> tensor<1x10xf16>
    // CHECK: [[HEAD_FLAT:%.+]] = IE.AffineReshape([[ROW]]) {{.*}} : tensor<1x10xf16> -> tensor<10xf16>
    // CHECK: [[LAST_ROW:%.+]] = IE.Slice [[ARG_0]] [11] [9] : tensor<21xf16> to tensor<9xf16>
    // CHECK: [[LAST_UPD:%.+]] = IE.Slice [[ARG_1]] [3] [3] : tensor<6xf16> to tensor<3xf16>
    // CHECK: [[LAST_GROUPS:%.+]] = IE.AffineReshape([[LAST_ROW]]) {{.*}} : tensor<9xf16> -> tensor<3x3xf16>
    // CHECK: [[LAST_KEEP:%.+]] = IE.Slice [[LAST_GROUPS]] [0, 1] [3, 2] : tensor<3x3xf16> to tensor<3x2xf16>
    // CHECK: [[LAST_UPD_2D:%.+]] = IE.AffineReshape([[LAST_UPD]]) {{.*}} : tensor<3xf16> -> tensor<3x1xf16>
    // CHECK: [[LAST_NEW:%.+]] = IE.Concat([[LAST_UPD_2D]], [[LAST_KEEP]]) {{.*}} : tensor<3x1xf16>, tensor<3x2xf16> -> tensor<3x3xf16>
    // CHECK: [[LAST_FLAT:%.+]] = IE.AffineReshape([[LAST_NEW]]) {{.*}} : tensor<3x3xf16> -> tensor<9xf16>
    // CHECK: [[TAIL:%.+]] = IE.Slice [[ARG_0]] [20] [1] : tensor<21xf16> to tensor<1xf16>
    // CHECK: [[OUT:%.+]] = IE.Concat([[HEAD]], [[HEAD_FLAT]], [[LAST_FLAT]], [[TAIL]]) {{.*}} : tensor<1xf16>, tensor<10xf16>, tensor<9xf16>, tensor<1xf16> -> tensor<21xf16>
    // CHECK: return [[OUT]] : tensor<21xf16>
}

// -----

// CHECK-LABEL: @MergeChainedScatterElementsUpdate
// CHECK-SAME: ([[DATA:%.+]]: tensor<1x64xi8>, [[IDX0:%.+]]: tensor<1x1xsi64>, [[IDX1:%.+]]: tensor<1x1xsi64>, [[IDX2:%.+]]: tensor<1x1xsi64>, [[IDX3:%.+]]: tensor<1x1xsi64>)
func.func @MergeChainedScatterElementsUpdate(%data: tensor<1x64xi8>, %idx0: tensor<1x1xsi64>, %idx1: tensor<1x1xsi64>, %idx2: tensor<1x1xsi64>, %idx3: tensor<1x1xsi64>) -> tensor<1x64xi8> {
    %axis = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %u0 = const.Declare tensor<1x1xi8> = dense<1> : tensor<1x1xi8>
    %u1 = const.Declare tensor<1x1xi8> = dense<2> : tensor<1x1xi8>
    %u2 = const.Declare tensor<1x1xi8> = dense<3> : tensor<1x1xi8>
    %u3 = const.Declare tensor<1x1xi8> = dense<4> : tensor<1x1xi8>

    %0 = IE.ScatterElementsUpdate(%data, %idx0, %u0, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>
    %1 = IE.ScatterElementsUpdate(%0, %idx1, %u1, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>
    %2 = IE.ScatterElementsUpdate(%1, %idx2, %u2, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>
    %3 = IE.ScatterElementsUpdate(%2, %idx3, %u3, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>

    return %3 : tensor<1x64xi8>

    // CHECK-DAG: [[MERGED_UPDATES:%.+]] = const.Declare tensor<1x4xi8> = dense<> : tensor<0xf80>, [#const.Concat<axis = <1 : i64>, offsets = <{{\[\[}}0, 0], [0, 1], [0, 2], [0, 3]]>, constants = <[dense<1> : tensor<1x1xi8>, dense<2> : tensor<1x1xi8>, dense<3> : tensor<1x1xi8>, dense<4> : tensor<1x1xi8>]>>]
    // CHECK: [[MERGED_INDICES:%.+]] = IE.Concat([[IDX0]], [[IDX1]], [[IDX2]], [[IDX3]])
    // CHECK-SAME: tensor<1x1xsi64>, tensor<1x1xsi64>, tensor<1x1xsi64>, tensor<1x1xsi64> -> tensor<1x4xsi64>
    // CHECK: [[RESULT:%.+]] = IE.ScatterElementsUpdate([[DATA]], [[MERGED_INDICES]], [[MERGED_UPDATES]]) {axis_value = 1 : i64, reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true}
    // CHECK-SAME: tensor<1x64xi8>, tensor<1x4xsi64>, tensor<1x4xi8> -> tensor<1x64xi8>
    // CHECK: return [[RESULT]] : tensor<1x64xi8>
}

// -----

// 1D stride-3 with no tail (bodyEnd == inputSize == 10), exercising the tailSize == 0 path.
// CHECK-LABEL: @ConvertScatterNDArithmeticIndivisible
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<10xf16>, [[ARG_1:%[^:]+]]: tensor<3xf16>
func.func @ConvertScatterNDArithmeticIndivisible(%arg0: tensor<10xf16>, %arg1: tensor<3xf16>) -> tensor<10xf16> {
    %cst = const.Declare tensor<3x1xsi64> = dense<[[1], [4], [7]]> : tensor<3x1xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<10xf16>, tensor<3x1xsi64>, tensor<3xf16> -> tensor<10xf16>
    return %0 : tensor<10xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK-NOT: IE.StridedSlice
    // CHECK: [[HEAD:%.+]] = IE.Slice [[ARG_0]] [0] [1] : tensor<10xf16> to tensor<1xf16>
    // CHECK: [[BODY:%.+]] = IE.Slice [[ARG_0]] [1] [9] : tensor<10xf16> to tensor<9xf16>
    // CHECK: [[GROUPS:%.+]] = IE.AffineReshape([[BODY]]) {{.*}} : tensor<9xf16> -> tensor<3x3xf16>
    // CHECK: [[KEEP:%.+]] = IE.Slice [[GROUPS]] [0, 1] [3, 2] : tensor<3x3xf16> to tensor<3x2xf16>
    // CHECK: [[UPD:%.+]] = IE.AffineReshape([[ARG_1]]) {{.*}} : tensor<3xf16> -> tensor<3x1xf16>
    // CHECK: [[NEW:%.+]] = IE.Concat([[UPD]], [[KEEP]]) {{.*}} : tensor<3x1xf16>, tensor<3x2xf16> -> tensor<3x3xf16>
    // CHECK: [[BODY_NEW:%.+]] = IE.AffineReshape([[NEW]]) {{.*}} : tensor<3x3xf16> -> tensor<9xf16>
    // CHECK: [[OUT:%.+]] = IE.Concat([[HEAD]], [[BODY_NEW]]) {{.*}} : tensor<1xf16>, tensor<9xf16> -> tensor<10xf16>
    // CHECK: return [[OUT]] : tensor<10xf16>
}

// -----

// Non-arithmetic indices (uneven gaps) — handled by the unified Select pattern.
// CHECK-LABEL: @ConvertScatterNDNonArithmeticIndicesViaSelect
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<15xf16>, [[ARG_1:%[^:]+]]: tensor<4xf16>
func.func @ConvertScatterNDNonArithmeticIndicesViaSelect(%arg0: tensor<15xf16>, %arg1: tensor<4xf16>) -> tensor<15xf16> {
    %cst = const.Declare tensor<4x1xsi64> = dense<[[1], [4], [7], [11]]> : tensor<4x1xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<15xf16>, tensor<4x1xsi64>, tensor<4xf16> -> tensor<15xf16>
    return %0 : tensor<15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[S0:%.+]] = IE.Slice [[ARG_0]] [0] [1] : tensor<15xf16> to tensor<1xf16>
    // CHECK: [[U0:%.+]] = IE.Slice [[ARG_1]] [0] [1] : tensor<4xf16> to tensor<1xf16>
    // CHECK: [[S1:%.+]] = IE.Slice [[ARG_0]] [2] [2] : tensor<15xf16> to tensor<2xf16>
    // CHECK: [[U1:%.+]] = IE.Slice [[ARG_1]] [1] [1] : tensor<4xf16> to tensor<1xf16>
    // CHECK: [[S2:%.+]] = IE.Slice [[ARG_0]] [5] [2] : tensor<15xf16> to tensor<2xf16>
    // CHECK: [[U2:%.+]] = IE.Slice [[ARG_1]] [2] [1] : tensor<4xf16> to tensor<1xf16>
    // CHECK: [[S3:%.+]] = IE.Slice [[ARG_0]] [8] [3] : tensor<15xf16> to tensor<3xf16>
    // CHECK: [[U3:%.+]] = IE.Slice [[ARG_1]] [3] [1] : tensor<4xf16> to tensor<1xf16>
    // CHECK: [[TAIL:%.+]] = IE.Slice [[ARG_0]] [12] [3] : tensor<15xf16> to tensor<3xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[S0]], [[U0]], [[S1]], [[U1]], [[S2]], [[U2]], [[S3]], [[U3]], [[TAIL]])
    // CHECK-SAME: -> tensor<15xf16>
    // CHECK: return [[CONCAT]] : tensor<15xf16>
}

// -----

// Do not merge when intermediate results have multiple uses
// CHECK-LABEL: @DoNotMergeScatterElementsMultiUse
// CHECK-SAME: ([[DATA:%.+]]: tensor<1x64xi8>
func.func @DoNotMergeScatterElementsMultiUse(%data: tensor<1x64xi8>, %idx0: tensor<1x1xsi64>, %idx1: tensor<1x1xsi64>) -> (tensor<1x64xi8>, tensor<1x64xi8>) {
    %axis = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %u0 = const.Declare tensor<1x1xi8> = dense<1> : tensor<1x1xi8>
    %u1 = const.Declare tensor<1x1xi8> = dense<2> : tensor<1x1xi8>

    %0 = IE.ScatterElementsUpdate(%data, %idx0, %u0, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>
    %1 = IE.ScatterElementsUpdate(%0, %idx1, %u1, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<1x64xi8>

    return %0, %1 : tensor<1x64xi8>, tensor<1x64xi8>

    // CHECK: IE.ScatterElementsUpdate([[DATA]]
    // CHECK: IE.ScatterElementsUpdate
    // CHECK: return
}

// -----

// CHECK-LABEL: @ConvertScatterNDInconsistentSecondRowViaSelect
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<25xf16>, [[ARG_1:%[^:]+]]: tensor<6xf16>
func.func @ConvertScatterNDInconsistentSecondRowViaSelect(%arg0: tensor<25xf16>, %arg1: tensor<6xf16>) -> tensor<25xf16> {
    %cst = const.Declare tensor<6x1xsi64> = dense<[[1], [4], [7], [11], [15], [19]]> : tensor<6x1xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<25xf16>, tensor<6x1xsi64>, tensor<6xf16> -> tensor<25xf16>
    return %0 : tensor<25xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[S0:%.+]] = IE.Slice [[ARG_0]] [0] [1] : tensor<25xf16> to tensor<1xf16>
    // CHECK: [[U0:%.+]] = IE.Slice [[ARG_1]] [0] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[S1:%.+]] = IE.Slice [[ARG_0]] [2] [2] : tensor<25xf16> to tensor<2xf16>
    // CHECK: [[U1:%.+]] = IE.Slice [[ARG_1]] [1] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[S2:%.+]] = IE.Slice [[ARG_0]] [5] [2] : tensor<25xf16> to tensor<2xf16>
    // CHECK: [[U2:%.+]] = IE.Slice [[ARG_1]] [2] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[S3:%.+]] = IE.Slice [[ARG_0]] [8] [3] : tensor<25xf16> to tensor<3xf16>
    // CHECK: [[U3:%.+]] = IE.Slice [[ARG_1]] [3] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[S4:%.+]] = IE.Slice [[ARG_0]] [12] [3] : tensor<25xf16> to tensor<3xf16>
    // CHECK: [[U4:%.+]] = IE.Slice [[ARG_1]] [4] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[S5:%.+]] = IE.Slice [[ARG_0]] [16] [3] : tensor<25xf16> to tensor<3xf16>
    // CHECK: [[U5:%.+]] = IE.Slice [[ARG_1]] [5] [1] : tensor<6xf16> to tensor<1xf16>
    // CHECK: [[TAIL:%.+]] = IE.Slice [[ARG_0]] [20] [5] : tensor<25xf16> to tensor<5xf16>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[S0]], [[U0]], [[S1]], [[U1]], [[S2]], [[U2]], [[S3]], [[U3]], [[S4]], [[U4]], [[S5]], [[U5]], [[TAIL]])
    // CHECK-SAME: -> tensor<25xf16>
    // CHECK: return [[CONCAT]] : tensor<25xf16>
}

// -----

// Point-level scatter with identity prefix selecting non-contiguous columns [0, 2] from last dim (size 3).
// CHECK-LABEL: @ConvertScatterNDUpdateSelectColumns
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<4x3xf32>, [[ARG_1:%[^:]+]]: tensor<4x2xf32>
func.func @ConvertScatterNDUpdateSelectColumns(%arg0: tensor<4x3xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x3xf32> {
    %cst = const.Declare tensor<4x2x2xsi64> = dense<[
        [[0, 0], [0, 2]],
        [[1, 0], [1, 2]],
        [[2, 0], [2, 2]],
        [[3, 0], [3, 2]]
    ]> : tensor<4x2x2xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x3xf32>, tensor<4x2x2xsi64>, tensor<4x2xf32> -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_UPD_0:%.+]] = IE.Slice [[ARG_1]] [0, 0] [4, 1] : tensor<4x2xf32> to tensor<4x1xf32>
    // CHECK: [[SLICE_INP_1:%.+]] = IE.Slice [[ARG_0]] [0, 1] [4, 1] : tensor<4x3xf32> to tensor<4x1xf32>
    // CHECK: [[SLICE_UPD_2:%.+]] = IE.Slice [[ARG_1]] [0, 1] [4, 1] : tensor<4x2xf32> to tensor<4x1xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_UPD_0]], [[SLICE_INP_1]], [[SLICE_UPD_2]])
    // CHECK-SAME: tensor<4x1xf32>, tensor<4x1xf32>, tensor<4x1xf32> -> tensor<4x3xf32>
    // CHECK: return [[CONCAT]] : tensor<4x3xf32>
}

// -----

// Partial scatter with identity prefix selecting non-contiguous rows [7, 14, 25, 33] from dim2 (size 34).
// Remaining dim3 (size 3) is bulk-copied from updates.
// CHECK-LABEL: @ConvertScatterNDUpdateSelectRows4D
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x4x34x3xf32>, [[ARG_1:%[^:]+]]: tensor<1x4x4x3xf32>
func.func @ConvertScatterNDUpdateSelectRows4D(%arg0: tensor<1x4x34x3xf32>, %arg1: tensor<1x4x4x3xf32>) -> tensor<1x4x34x3xf32> {
    %cst = const.Declare tensor<1x4x4x3xsi64> = dense<[
        [[[0, 0, 7], [0, 0, 14], [0, 0, 25], [0, 0, 33]],
         [[0, 1, 7], [0, 1, 14], [0, 1, 25], [0, 1, 33]],
         [[0, 2, 7], [0, 2, 14], [0, 2, 25], [0, 2, 33]],
         [[0, 3, 7], [0, 3, 14], [0, 3, 25], [0, 3, 33]]]
    ]> : tensor<1x4x4x3xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x34x3xf32>, tensor<1x4x4x3xsi64>, tensor<1x4x4x3xf32> -> tensor<1x4x34x3xf32>
    return %0 : tensor<1x4x34x3xf32>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_D0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0] [1, 4, 7, 3] : tensor<1x4x34x3xf32> to tensor<1x4x7x3xf32>
    // CHECK: [[SLICE_U0:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0] [1, 4, 1, 3] : tensor<1x4x4x3xf32> to tensor<1x4x1x3xf32>
    // CHECK: [[SLICE_D1:%.+]] = IE.Slice [[ARG_0]] [0, 0, 8, 0] [1, 4, 6, 3] : tensor<1x4x34x3xf32> to tensor<1x4x6x3xf32>
    // CHECK: [[SLICE_U1:%.+]] = IE.Slice [[ARG_1]] [0, 0, 1, 0] [1, 4, 1, 3] : tensor<1x4x4x3xf32> to tensor<1x4x1x3xf32>
    // CHECK: [[SLICE_D2:%.+]] = IE.Slice [[ARG_0]] [0, 0, 15, 0] [1, 4, 10, 3] : tensor<1x4x34x3xf32> to tensor<1x4x10x3xf32>
    // CHECK: [[SLICE_U2:%.+]] = IE.Slice [[ARG_1]] [0, 0, 2, 0] [1, 4, 1, 3] : tensor<1x4x4x3xf32> to tensor<1x4x1x3xf32>
    // CHECK: [[SLICE_D3:%.+]] = IE.Slice [[ARG_0]] [0, 0, 26, 0] [1, 4, 7, 3] : tensor<1x4x34x3xf32> to tensor<1x4x7x3xf32>
    // CHECK: [[SLICE_U3:%.+]] = IE.Slice [[ARG_1]] [0, 0, 3, 0] [1, 4, 1, 3] : tensor<1x4x4x3xf32> to tensor<1x4x1x3xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_D0]], [[SLICE_U0]], [[SLICE_D1]], [[SLICE_U1]], [[SLICE_D2]], [[SLICE_U2]], [[SLICE_D3]], [[SLICE_U3]])
    // CHECK-SAME: -> tensor<1x4x34x3xf32>
    // CHECK: return [[CONCAT]] : tensor<1x4x34x3xf32>
}

// -----

// Partial scatter 5D: identity prefix selects rows [7, 14, 25, 33] in dim2; remaining dims [3, 3] bulk-copied.
// CHECK-LABEL: @ConvertScatterNDUpdateSelectRows5D
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<1x4x34x3x3xf32>, [[ARG_1:%[^:]+]]: tensor<1x4x4x3x3xf32>
func.func @ConvertScatterNDUpdateSelectRows5D(%arg0: tensor<1x4x34x3x3xf32>, %arg1: tensor<1x4x4x3x3xf32>) -> tensor<1x4x34x3x3xf32> {
    %cst = const.Declare tensor<1x4x4x3xsi64> = dense<[
        [[[0, 0, 7], [0, 0, 14], [0, 0, 25], [0, 0, 33]],
         [[0, 1, 7], [0, 1, 14], [0, 1, 25], [0, 1, 33]],
         [[0, 2, 7], [0, 2, 14], [0, 2, 25], [0, 2, 33]],
         [[0, 3, 7], [0, 3, 14], [0, 3, 25], [0, 3, 33]]]
    ]> : tensor<1x4x4x3xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x4x34x3x3xf32>, tensor<1x4x4x3xsi64>, tensor<1x4x4x3x3xf32> -> tensor<1x4x34x3x3xf32>
    return %0 : tensor<1x4x34x3x3xf32>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_D0:%.+]] = IE.Slice [[ARG_0]] [0, 0, 0, 0, 0] [1, 4, 7, 3, 3] : tensor<1x4x34x3x3xf32> to tensor<1x4x7x3x3xf32>
    // CHECK: [[SLICE_U0:%.+]] = IE.Slice [[ARG_1]] [0, 0, 0, 0, 0] [1, 4, 1, 3, 3] : tensor<1x4x4x3x3xf32> to tensor<1x4x1x3x3xf32>
    // CHECK: [[SLICE_D1:%.+]] = IE.Slice [[ARG_0]] [0, 0, 8, 0, 0] [1, 4, 6, 3, 3] : tensor<1x4x34x3x3xf32> to tensor<1x4x6x3x3xf32>
    // CHECK: [[SLICE_U1:%.+]] = IE.Slice [[ARG_1]] [0, 0, 1, 0, 0] [1, 4, 1, 3, 3] : tensor<1x4x4x3x3xf32> to tensor<1x4x1x3x3xf32>
    // CHECK: [[SLICE_D2:%.+]] = IE.Slice [[ARG_0]] [0, 0, 15, 0, 0] [1, 4, 10, 3, 3] : tensor<1x4x34x3x3xf32> to tensor<1x4x10x3x3xf32>
    // CHECK: [[SLICE_U2:%.+]] = IE.Slice [[ARG_1]] [0, 0, 2, 0, 0] [1, 4, 1, 3, 3] : tensor<1x4x4x3x3xf32> to tensor<1x4x1x3x3xf32>
    // CHECK: [[SLICE_D3:%.+]] = IE.Slice [[ARG_0]] [0, 0, 26, 0, 0] [1, 4, 7, 3, 3] : tensor<1x4x34x3x3xf32> to tensor<1x4x7x3x3xf32>
    // CHECK: [[SLICE_U3:%.+]] = IE.Slice [[ARG_1]] [0, 0, 3, 0, 0] [1, 4, 1, 3, 3] : tensor<1x4x4x3x3xf32> to tensor<1x4x1x3x3xf32>
    // CHECK: [[CONCAT:%.+]] = IE.Concat([[SLICE_D0]], [[SLICE_U0]], [[SLICE_D1]], [[SLICE_U1]], [[SLICE_D2]], [[SLICE_U2]], [[SLICE_D3]], [[SLICE_U3]])
    // CHECK-SAME: -> tensor<1x4x34x3x3xf32>
    // CHECK: return [[CONCAT]] : tensor<1x4x34x3x3xf32>
}

// -----

// Negative case: identity prefix does NOT hold (last row has wrong spatial coords).
// CHECK-LABEL: @DoNotConvertSelectNonIdentityPrefix
// CHECK-SAME:    [[ARG_0:%[^:]+]]: tensor<4x3xf32>, [[ARG_1:%[^:]+]]: tensor<4x2xf32>
func.func @DoNotConvertSelectNonIdentityPrefix(%arg0: tensor<4x3xf32>, %arg1: tensor<4x2xf32>) -> tensor<4x3xf32> {
    %cst = const.Declare tensor<4x2x2xsi64> = dense<[
        [[0, 0], [0, 2]],
        [[1, 0], [1, 2]],
        [[2, 0], [2, 2]],
        [[0, 0], [0, 2]]
    ]> : tensor<4x2x2xsi64>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x3xf32>, tensor<4x2x2xsi64>, tensor<4x2xf32> -> tensor<4x3xf32>
    return %0 : tensor<4x3xf32>

    // CHECK: [[RESULT:%.+]] = IE.ScatterNDUpdate([[ARG_0]], {{.*}}, [[ARG_1]])
    // CHECK: return [[RESULT]]
}

// -----

// Do not merge when reduction types differ
// CHECK-LABEL: @DoNotMergeScatterElementsDifferentReduction
// CHECK-SAME: ([[DATA:%.+]]: tensor<1x64xf16>
func.func @DoNotMergeScatterElementsDifferentReduction(%data: tensor<1x64xf16>, %idx0: tensor<1x1xsi64>, %idx1: tensor<1x1xsi64>) -> tensor<1x64xf16> {
    %axis = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %u0 = const.Declare tensor<1x1xf16> = dense<1.0> : tensor<1x1xf16>
    %u1 = const.Declare tensor<1x1xf16> = dense<2.0> : tensor<1x1xf16>

    %0 = IE.ScatterElementsUpdate(%data, %idx0, %u0, %axis) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<1x64xf16>, tensor<1x1xsi64>, tensor<1x1xf16>, tensor<si64> -> tensor<1x64xf16>
    %1 = IE.ScatterElementsUpdate(%0, %idx1, %u1, %axis) {reduction = #IE.scatter_elements_update_reduction_type<SUM>, use_init_val = true} : tensor<1x64xf16>, tensor<1x1xsi64>, tensor<1x1xf16>, tensor<si64> -> tensor<1x64xf16>

    return %1 : tensor<1x64xf16>

    // CHECK: IE.ScatterElementsUpdate([[DATA]]
    // CHECK: IE.ScatterElementsUpdate
    // CHECK: return
}

// -----

// Do not merge when axes differ between ops in the chain
// CHECK-LABEL: @DoNotMergeScatterElementsDifferentAxis
// CHECK-SAME: ([[DATA:%.+]]: tensor<4x64xi8>
func.func @DoNotMergeScatterElementsDifferentAxis(%data: tensor<4x64xi8>, %idx0: tensor<1x1xsi64>, %idx1: tensor<1x1xsi64>) -> tensor<4x64xi8> {
    %axis0 = const.Declare tensor<si64> = dense<0> : tensor<si64>
    %axis1 = const.Declare tensor<si64> = dense<1> : tensor<si64>
    %u0 = const.Declare tensor<1x1xi8> = dense<1> : tensor<1x1xi8>
    %u1 = const.Declare tensor<1x1xi8> = dense<2> : tensor<1x1xi8>

    %0 = IE.ScatterElementsUpdate(%data, %idx0, %u0, %axis0) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<4x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<4x64xi8>
    %1 = IE.ScatterElementsUpdate(%0, %idx1, %u1, %axis1) {reduction = #IE.scatter_elements_update_reduction_type<NONE>, use_init_val = true} : tensor<4x64xi8>, tensor<1x1xsi64>, tensor<1x1xi8>, tensor<si64> -> tensor<4x64xi8>

    return %1 : tensor<4x64xi8>

    // CHECK: IE.ScatterElementsUpdate([[DATA]]
    // CHECK: IE.ScatterElementsUpdate
    // CHECK: return
}
