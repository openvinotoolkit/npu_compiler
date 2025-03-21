//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-scatterndupdate-to-strided-concat
// --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// CHECK-LABEL: @ConvertScatterNDUpdateToStridedConcat
func.func @ConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,12]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-NOT: IE.ScatterNDUpdate
    // CHECK: [[SLICE_1:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 1], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[SLICE_2:%.*]] = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 2], ellipsis_mask = [0, 0, 0, 0, 0], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 1, 1, 1, 15], new_axis_mask = [0, 0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0, 0], strides_attr = [1, 1, 1, 1, 3]} : tensor<1x1x1x1x15xf16> -> tensor<1x1x1x1x5xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE_1]], [[SLICE_2]]) {per_axis = #IE.Concat<axis = 4 : i64, offset = 1 : i64, stride = 3 : i64>} : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[CONCAT]] : tensor<1x1x1x1x15xf16>
}

// -----

// The last dim value is [0,0,0,0,11], so it will remain IE.ScatterNDUpdate.
// CHECK-LABEL: @DoNotConvertScatterNDUpdateToStridedConcat
func.func @DoNotConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9],[0,0,0,0,11]]]]]]> : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1x5x5xsi32> =
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate(%arg0, [[CST]], %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[RESULT]] : tensor<1x1x1x1x15xf16>
}

// -----

// The indices shape could not meet Integer stride condition, so it will remain IE.ScatterNDUpdate.
// CHECK-LABEL: @NotIntegerStrideDoNotConvertScatterNDUpdateToStridedConcat
func.func @NotIntegerStrideDoNotConvertScatterNDUpdateToStridedConcat(%arg0:  tensor<1x1x1x1x15xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x15xf16>{
    %cst = const.Declare tensor<1x1x1x1x4x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,3],[0,0,0,0,6],[0,0,0,0,9]]]]]]> : tensor<1x1x1x1x4x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x4x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>

    return %0 : tensor<1x1x1x1x15xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x1x1x1x4x5xsi32> =
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate(%arg0, [[CST]], %arg1) : tensor<1x1x1x1x15xf16>, tensor<1x1x1x1x4x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x15xf16>
    // CHECK: return [[RESULT]] : tensor<1x1x1x1x15xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithSameSize
func.func @ConvertToSliceConcatElementsUpdateWithSameSize(%arg0:  tensor<1x1x1x1x5xf16>, %arg1 : tensor<1x1x1x1x5xf16> ) -> tensor<1x1x1x1x5xf16>{
    %cst = const.Declare tensor<1x1x1x1x5x5xsi32> = dense<[[[[[[0,0,0,0,0],[0,0,0,0,1],[0,0,0,0,2],[0,0,0,0,3],[0,0,0,0,4]]]]]]>  : tensor<1x1x1x1x5x5xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1x5xf16>, tensor<1x1x1x1x5x5xsi32>, tensor<1x1x1x1x5xf16> -> tensor<1x1x1x1x5xf16>

    return %0 : tensor<1x1x1x1x5xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return %arg1 : tensor<1x1x1x1x5xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithOneElement
func.func @ConvertToSliceConcatElementsUpdateWithOneElement(%arg0:  tensor<1x1x1x1xf16>, %arg1 : tensor<1x1x1x1xf16> ) -> tensor<1x1x1x1xf16>{
    %cst = const.Declare tensor<1x1x1x1x4xsi32> = dense<[[[[[0,0,0,0]]]]]>  : tensor<1x1x1x1x4xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x1x1x1xf16>, tensor<1x1x1x1x4xsi32>, tensor<1x1x1x1xf16> -> tensor<1x1x1x1xf16>

    return %0 : tensor<1x1x1x1xf16>

    // CHECK-NOT: ScatterNDUpdate
    // CHECK: return %arg1 : tensor<1x1x1x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithReplaceOneElement
func.func @ConvertToSliceConcatElementsUpdateWithReplaceOneElement(%arg0:  tensor<1x10x1xf16>, %arg1 : tensor<1x1x1xf16> ) -> tensor<1x10x1xf16>{
    %cst = const.Declare tensor<1x1x1x3xsi32> = dense<0> : tensor<1x1x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x10x1xf16>, tensor<1x1x1x3xsi32>, tensor<1x1x1xf16> -> tensor<1x10x1xf16>

    return %0 : tensor<1x10x1xf16>

    // CHECK: [[SLICE:%.*]] = IE.Slice %arg0 [0, 1, 0] [1, 9, 1] : tensor<1x10x1xf16> to tensor<1x9x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 1, 0]]} : tensor<1x1x1xf16>, tensor<1x9x1xf16> -> tensor<1x10x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x10x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithTwoSlice
func.func @ConvertToSliceConcatElementsUpdateWithTwoSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 249, 0]], [[0, 250, 0]], [[0, 251, 0]], [[0, 252, 0]], [[0, 253, 0]], [[0, 254, 0]], [[0, 255, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 249, 1] : tensor<1x326x1xf16> to tensor<1x249x1xf16>
    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [0, 256, 0] [1, 70, 1] : tensor<1x326x1xf16> to tensor<1x70x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 249, 0], [0, 256, 0]]} : tensor<1x249x1xf16>, tensor<1x7x1xf16>, tensor<1x70x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithRightSlice
func.func @ConvertToSliceConcatElementsUpdateWithRightSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 0, 0]], [[0, 1, 0]], [[0, 2, 0]], [[0, 3, 0]], [[0, 4, 0]], [[0, 5, 0]], [[0, 6, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [0, 7, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat(%arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 7, 0]]} : tensor<1x7x1xf16>, tensor<1x319x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatElementsUpdateWithLeftSlice
func.func @ConvertToSliceConcatElementsUpdateWithLeftSlice(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 319, 0]], [[0, 320, 0]], [[0, 321, 0]], [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 319, 1] : tensor<1x326x1xf16> to tensor<1x319x1xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1) {static_offsets = {{\[\[}}0, 0, 0], [0, 319, 0]]} : tensor<1x319x1xf16>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>
    // CHECK: return [[CONCAT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSliceConcatElementsUpdateWithIllegalIndicesData
func.func @NotConvertToSliceConcatElementsUpdateWithIllegalIndicesData(%arg0:  tensor<1x326x1xf16>, %arg1 : tensor<1x7x1xf16> ) -> tensor<1x326x1xf16>{
    %cst = const.Declare tensor<1x7x1x3xsi32> = dense<[[[[0, 319, 0]], [[0, 320, 0]], [[0, 100, 0]], [[0, 322, 0]], [[0, 323, 0]], [[0, 324, 0]], [[0, 325, 0]]]]> : tensor<1x7x1x3xsi64>, [#const.CastElemType<si32>]
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<1x326x1xf16>, tensor<1x7x1x3xsi32>, tensor<1x7x1xf16> -> tensor<1x326x1xf16>

    return %0 : tensor<1x326x1xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate
    // CHECK: return [[RESULT]] : tensor<1x326x1xf16>
}

// -----

// CHECK-LABEL: @ConvertToSliceConcatTensorUpdate
func.func @ConvertToSliceConcatTensorUpdate(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[1], [2]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK: [[SLICE_LEFT:%.*]] = IE.Slice %arg0 [0, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[SLICE_RIGHT:%.*]] = IE.Slice %arg0 [3, 0, 0] [1, 4, 4] : tensor<4x4x4xf16> to tensor<1x4x4xf16>
    // CHECK: [[CONCAT:%.*]] = IE.Concat([[SLICE_LEFT]], %arg1, [[SLICE_RIGHT]]) {static_offsets = {{\[\[}}0, 0, 0], [1, 0, 0], [3, 0, 0]]} : tensor<1x4x4xf16>, tensor<2x4x4xf16>, tensor<1x4x4xf16> -> tensor<4x4x4xf16>
    // CHECK: return [[CONCAT]] : tensor<4x4x4xf16>
}

// -----

// CHECK-LABEL: @NotConvertToSliceConcatTensorUpdateWithIllegalIndicesData
func.func @NotConvertToSliceConcatTensorUpdateWithIllegalIndicesData(%arg0:  tensor<4x4x4xf16>, %arg1 : tensor<2x4x4xf16> ) -> tensor<4x4x4xf16>{
    %cst = const.Declare tensor<2x1xsi32> = dense<[[2], [1]]> : tensor<2x1xsi32>
    %0 = IE.ScatterNDUpdate(%arg0, %cst, %arg1) : tensor<4x4x4xf16>, tensor<2x1xsi32>, tensor<2x4x4xf16> -> tensor<4x4x4xf16>

    return %0 : tensor<4x4x4xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare
    // CHECK: [[RESULT:%.*]] = IE.ScatterNDUpdate
    // CHECK: return [[RESULT]] : tensor<4x4x4xf16>
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
