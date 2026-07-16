//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-gather %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @ConvertGatherToSliceAxis0
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<18x8x72x64xf16>)
func.func @ConvertGatherToSliceAxis0(%arg0: tensor<18x8x72x64xf16>) -> tensor<8x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<9> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<8x72x64xf16>

    return %0 : tensor<8x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [9, 0, 0, 0] [1, 8, 72, 64] : tensor<18x8x72x64xf16> to tensor<1x8x72x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [8, 72, 64]} : tensor<1x8x72x64xf16> -> tensor<8x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// -----

// CHECK-LABEL: @ConvertGatherToSliceAxis1
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<18x8x72x64xf16>)
func.func @ConvertGatherToSliceAxis1(%arg0: tensor<18x8x72x64xf16>) -> tensor<18x72x64xf16> {
    %cst = const.Declare tensor<si32> = dense<3> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<18x8x72x64xf16>, tensor<si32> -> tensor<18x72x64xf16>

    return %0 : tensor<18x72x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 3, 0, 0] [18, 1, 72, 64] : tensor<18x8x72x64xf16> to tensor<18x1x72x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [18, 72, 64]} : tensor<18x1x72x64xf16> -> tensor<18x72x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @ConvertGatherToSlicewith3DShape
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<8x72x64xf16>)
func.func @ConvertGatherToSlicewith3DShape(%arg0: tensor<8x72x64xf16>) -> tensor<8x64xf16> {
    %cst = const.Declare tensor<si32> = dense<8> : tensor<si32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<8x72x64xf16>, tensor<si32> -> tensor<8x64xf16>

    return %0 : tensor<8x64xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[SLICE:%.+]] = IE.Slice [[ARG_0]] [0, 8, 0] [8, 1, 64] : tensor<8x72x64xf16> to tensor<8x1x64xf16>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SLICE]]) {shape_value = [8, 64]} : tensor<8x1x64xf16> -> tensor<8x64xf16>
    // CHECK:       return [[RESHAPE]]
}

// CHECK-LABEL: @CannotConvertGatherToSlice
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1xf32>, [[ARG_1:%[^:]+]]: tensor<1x8x16x16xf16>)
func.func @CannotConvertGatherToSlice(%arg0: tensor<1xf32>, %arg1: tensor<1x8x16x16xf16>) -> tensor<1x1x16x16xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    %1 = IE.Gather(%arg1, %0) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>

    return %1 : tensor<1x1x16x16xf16>

    // CHECK:       [[CONVERT:%.+]] = IE.Convert([[ARG_0]]) {dstElemType = si32} : tensor<1xf32> -> tensor<1xsi32>
    // CHECK:       [[GATHER:%.+]] = IE.Gather([[ARG_1]], [[CONVERT]]) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x8x16x16xf16>, tensor<1xsi32> -> tensor<1x1x16x16xf16>
    // CHECK:       return [[GATHER]] : tensor<1x1x16x16xf16>
}

// -----

// CHECK-LABEL: @ConvertGatherToReverseWithReverseContiguousIndices
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<16x224x224x3xf16>
func.func @ConvertGatherToReverseWithReverseContiguousIndices(%arg0: tensor<16x224x224x3xf16>) -> tensor<16x224x224x3xf16> {
    %cst = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 3 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<16x224x224x3xf16>, tensor<3xsi32> -> tensor<16x224x224x3xf16>

    return %0 : tensor<16x224x224x3xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[REVERSE:%.+]] = IE.Reverse([[INPUT]]) {axis_value = [3], mode = #IE.reverse_mode<INDEX>} : tensor<16x224x224x3xf16> -> tensor<16x224x224x3xf16>
    // CHECK:       return [[REVERSE]] : tensor<16x224x224x3xf16>
}

// -----

// CHECK-LABEL: @NotConvertGatherToReverseWithIndicesFirst
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x3x640x640xf16>
func.func @NotConvertGatherToReverseWithIndicesFirst(%arg0: tensor<1x3x640x640xf16>) -> tensor<1x3x640x640xf16> {
    %cst = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1x3x640x640xf16>, tensor<3xsi32> -> tensor<1x3x640x640xf16>

    return %0 : tensor<1x3x640x640xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<3xsi32> = dense<[2, 1, 0]> : tensor<3xsi32>
    // CHECK:       [[Gather:%.+]] = IE.Gather([[INPUT]], [[CST]]) {axis_value = 1 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<1x3x640x640xf16>, tensor<3xsi32> -> tensor<1x3x640x640xf16>
    // CHECK:       return [[Gather]] : tensor<1x3x640x640xf16>
}

// -----

// CHECK-LABEL: @FuseBroadcastGatherND
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x16x1x8xf16>
func.func @FuseBroadcastGatherND(%arg0: tensor<1x1x16x1x8xf16>) -> tensor<4x16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<4x2xsi32> = dense<[[0, 0], [0, 1], [0, 2], [0, 3]]> : tensor<4x2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<1x1x16x1x8xf16>, tensor<5xsi64> -> tensor<1x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<4x2xsi32> -> tensor<4x16x1x8xf16>
    return %result : tensor<4x16x1x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
    // CHECK-SAME:    tensor<1x1x16x1x8xf16> -> tensor<16x1x8xf16>
    // CHECK:     [[BCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:    tensor<16x1x8xf16>
    // CHECK-SAME:    tensor<4x16x1x8xf16>
    // CHECK:     return [[BCAST]] : tensor<4x16x1x8xf16>
}

// -----

// CHECK-LABEL: @FuseBroadcastGatherNDHighRankIndices
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x16x8xf16>
func.func @FuseBroadcastGatherNDHighRankIndices(%arg0: tensor<1x1x1x16x8xf16>) -> tensor<3x4x2x16x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 6, 16, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<3x4x2x3xsi32> = dense<0> : tensor<3x4x2x3xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<1x1x1x16x8xf16>, tensor<5xsi64> -> tensor<1x4x6x16x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x6x16x8xf16>, tensor<3x4x2x3xsi32> -> tensor<3x4x2x16x8xf16>
    return %result : tensor<3x4x2x16x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
    // CHECK-SAME:    tensor<1x1x1x16x8xf16> -> tensor<16x8xf16>
    // CHECK:     [[BCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:    tensor<16x8xf16>
    // CHECK-SAME:    tensor<3x4x2x16x8xf16>
    // CHECK:     return [[BCAST]] : tensor<3x4x2x16x8xf16>
}

// -----

// CHECK-LABEL: @FoldBroadcastAxisOutsideLastDim
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x1x1x8xf16>
func.func @FoldBroadcastAxisOutsideLastDim(%arg0: tensor<1x1x1x1x8xf16>) -> tensor<4x16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<4x2xsi32> = dense<[[0, 0], [0, 1], [0, 2], [0, 3]]> : tensor<4x2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<1x1x1x1x8xf16>, tensor<5xsi64> -> tensor<1x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<4x2xsi32> -> tensor<4x16x1x8xf16>
    return %result : tensor<4x16x1x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
    // CHECK-SAME:    tensor<1x1x1x1x8xf16> -> tensor<1x1x8xf16>
    // CHECK:     [[BCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:    tensor<1x1x8xf16>
    // CHECK-SAME:    tensor<4x16x1x8xf16>
    // CHECK:     return [[BCAST]] : tensor<4x16x1x8xf16>
}

// -----

// CHECK-LABEL: @FuseBroadcastGatherNDPartialSlices
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x16x1x8xf16>
func.func @FuseBroadcastGatherNDPartialSlices(%arg0: tensor<1x1x16x1x8xf16>) -> tensor<3x16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<3x2xsi32> = dense<[[0, 0], [0, 1], [0, 2]]> : tensor<3x2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<1x1x16x1x8xf16>, tensor<5xsi64> -> tensor<1x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<3x2xsi32> -> tensor<3x16x1x8xf16>
    return %result : tensor<3x16x1x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
    // CHECK-SAME:    tensor<1x1x16x1x8xf16> -> tensor<16x1x8xf16>
    // CHECK:     [[BCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:    tensor<16x1x8xf16>
    // CHECK-SAME:    tensor<3x16x1x8xf16>
    // CHECK:     return [[BCAST]] : tensor<3x16x1x8xf16>
}

// -----

// CHECK-LABEL: @FuseBroadcastGatherNDIndicesRank1
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x1x16x1x8xf16>
func.func @FuseBroadcastGatherNDIndicesRank1(%arg0: tensor<1x1x16x1x8xf16>) -> tensor<16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<2xsi32> = dense<[0, 1]> : tensor<2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<1x1x16x1x8xf16>, tensor<5xsi64> -> tensor<1x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<2xsi32> -> tensor<16x1x8xf16>
    return %result : tensor<16x1x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK:     [[RESHAPE:%.+]] = IE.Reshape([[INPUT]])
    // CHECK-SAME:    tensor<1x1x16x1x8xf16> -> tensor<16x1x8xf16>
    // CHECK:     return [[RESHAPE]] : tensor<16x1x8xf16>
}

// -----

// CHECK-LABEL: @FuseBroadcastGatherNDRankMismatch
// CHECK-SAME:  [[INPUT:%.+]]: tensor<16x1x8xf16>
func.func @FuseBroadcastGatherNDRankMismatch(%arg0: tensor<16x1x8xf16>) -> tensor<4x16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[1, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<4x2xsi32> = dense<[[0, 0], [0, 1], [0, 2], [0, 3]]> : tensor<4x2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<16x1x8xf16>, tensor<5xsi64> -> tensor<1x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<4x2xsi32> -> tensor<4x16x1x8xf16>
    return %result : tensor<4x16x1x8xf16>

    // CHECK-NOT: IE.GatherND
    // CHECK-NOT: IE.Reshape
    // CHECK:     [[BCAST:%.+]] = IE.Broadcast([[INPUT]],
    // CHECK-SAME:    tensor<16x1x8xf16>
    // CHECK-SAME:    tensor<4x16x1x8xf16>
    // CHECK:     return [[BCAST]] : tensor<4x16x1x8xf16>
}

// -----

// CHECK-LABEL: @NoFoldNonUnitBroadcastDim
func.func @NoFoldNonUnitBroadcastDim(%arg0: tensor<2x1x16x1x8xf16>) -> tensor<4x16x1x8xf16> {
    %cst_shape   = const.Declare tensor<5xsi64> = dense<[2, 4, 16, 1, 8]> : tensor<5xsi64>
    %cst_indices = const.Declare tensor<4x2xsi32> = dense<[[0, 0], [0, 1], [0, 2], [0, 3]]> : tensor<4x2xsi32>

    %broadcast = IE.Broadcast(%arg0, %cst_shape) {mode = #IE.broadcast_type<NUMPY>}
                   : tensor<2x1x16x1x8xf16>, tensor<5xsi64> -> tensor<2x4x16x1x8xf16>
    %result = IE.GatherND(%broadcast, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<2x4x16x1x8xf16>, tensor<4x2xsi32> -> tensor<4x16x1x8xf16>
    return %result : tensor<4x16x1x8xf16>

    // CHECK: IE.GatherND
}

// -----

// CHECK-LABEL: @NoFoldNoBroadcastInput
func.func @NoFoldNoBroadcastInput(%arg0: tensor<1x4x16x1x8xf16>) -> tensor<4x16x1x8xf16> {
    %cst_indices = const.Declare tensor<4x2xsi32> = dense<[[0, 0], [0, 1], [0, 2], [0, 3]]> : tensor<4x2xsi32>
    %result = IE.GatherND(%arg0, %cst_indices) {batch_dims = 0 : i64}
                   : tensor<1x4x16x1x8xf16>, tensor<4x2xsi32> -> tensor<4x16x1x8xf16>
    return %result : tensor<4x16x1x8xf16>

    // CHECK: IE.GatherND
}

// -----

// CHECK-LABEL: @ConvertRepeatInterleave2ToBroadcast
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x1024x64xf16>)
func.func @ConvertRepeatInterleave2ToBroadcast(%arg0: tensor<1x1x1024x64xf16>) -> tensor<1x1x1024x128xf16> {
    %cst = const.Declare tensor<128xsi32> = dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63]> : tensor<128xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 3 : i64, batch_dims = 0 : i64} : tensor<1x1x1024x64xf16>, tensor<128xsi32> -> tensor<1x1x1024x128xf16>
    return %0 : tensor<1x1x1024x128xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [1, 1, 1024, 64, 1]} : tensor<1x1x1024x64xf16> -> tensor<1x1x1024x64x1xf16>
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1x1024x64x1xf16>
    // CHECK-SAME:      -> tensor<1x1x1024x64x2xf16>
    // CHECK:       [[OUT:%.+]] = IE.Reshape([[BROADCAST]]) {shape_value = [1, 1, 1024, 128]} : tensor<1x1x1024x64x2xf16> -> tensor<1x1x1024x128xf16>
    // CHECK:       return [[OUT]]
}

// -----

// CHECK-LABEL: @ConvertRepeatInterleave3ToBroadcast
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<2x4x8xf16>)
func.func @ConvertRepeatInterleave3ToBroadcast(%arg0: tensor<2x4x8xf16>) -> tensor<2x4x24xf16> {
    %cst = const.Declare tensor<24xsi32> = dense<[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]> : tensor<24xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 2 : i64, batch_dims = 0 : i64} : tensor<2x4x8xf16>, tensor<24xsi32> -> tensor<2x4x24xf16>
    return %0 : tensor<2x4x24xf16>

    // CHECK-NOT:   IE.Gather
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[ARG_0]]) {shape_value = [2, 4, 8, 1]} : tensor<2x4x8xf16> -> tensor<2x4x8x1xf16>
    // CHECK:       [[BROADCAST:%.+]] = IE.Broadcast([[RESHAPE]],
    // CHECK-SAME:      {mode = #IE.broadcast_type<NUMPY>}
    // CHECK-SAME:      -> tensor<2x4x8x3xf16>
    // CHECK:       [[OUT:%.+]] = IE.Reshape([[BROADCAST]]) {shape_value = [2, 4, 24]} : tensor<2x4x8x3xf16> -> tensor<2x4x24xf16>
    // CHECK:       return [[OUT]]
}

// -----

// CHECK-LABEL: @NoConvertRepeatInterleaveNonUniformIndices
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x4xf16>)
func.func @NoConvertRepeatInterleaveNonUniformIndices(%arg0: tensor<1x1x4x4xf16>) -> tensor<1x1x4x8xf16> {
    %cst = const.Declare tensor<8xsi32> = dense<[0, 0, 1, 1, 2, 2, 3, 2]> : tensor<8xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 3 : i64, batch_dims = 0 : i64} : tensor<1x1x4x4xf16>, tensor<8xsi32> -> tensor<1x1x4x8xf16>
    return %0 : tensor<1x1x4x8xf16>

    // CHECK:       IE.Gather
}

// -----

// CHECK-LABEL: @NoConvertRepeatInterleaveIndivisible
// CHECK-SAME:    ([[ARG_0:%[^:]+]]: tensor<1x1x4x4xf16>)
func.func @NoConvertRepeatInterleaveIndivisible(%arg0: tensor<1x1x4x4xf16>) -> tensor<1x1x4x7xf16> {
    %cst = const.Declare tensor<7xsi32> = dense<[0, 0, 1, 1, 2, 2, 3]> : tensor<7xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 3 : i64, batch_dims = 0 : i64} : tensor<1x1x4x4xf16>, tensor<7xsi32> -> tensor<1x1x4x7xf16>
    return %0 : tensor<1x1x4x7xf16>

    // CHECK:       IE.Gather
}
