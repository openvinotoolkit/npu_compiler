//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-layers %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @MergeParallelMultiplyLayers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x512xf16>
func.func @MergeParallelMultiplyLayers(%arg0: tensor<1x1x512xf16>) -> (tensor<1x1x256xf16>, tensor<1x1x256xf16>) {
    %cst_0 = const.Declare tensor<1x1x1xf16> = dense<0.2> : tensor<1x1x1xf32> isSplat, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1xf16> = dense<0.3> : tensor<1x1x1xf32> isSplat, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf16>, tensor<1x1x1xf16> -> tensor<1x1x256xf16>

    %2 = IE.Slice %arg0 [0, 0, 256] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    %3 = IE.Multiply(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf16>, tensor<1x1x1xf16> -> tensor<1x1x256xf16>

    return %1, %3: tensor<1x1x256xf16>, tensor<1x1x256xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e-01> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e-01> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 2, 256]} : tensor<1x1x512xf16> -> tensor<1x1x2x256xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x1xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[LHS]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x256xf16>, tensor<1x1x2x1xf16> -> tensor<1x1x2x256xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 1, 512]} : tensor<1x1x2x256xf16> -> tensor<1x1x512xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 256] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x1x256xf16>, tensor<1x1x256xf16>
}

// -----

// CHECK-LABEL: @MergeParallelMultiplyLayersWithDiffRanks
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x512xf16>
func.func @MergeParallelMultiplyLayersWithDiffRanks(%arg0: tensor<1x1x512xf16>) -> (tensor<1x1x256xf16>, tensor<1x1x256xf16>) {
    %cst_0 = const.Declare tensor<1x1xf16> = dense<0.2> : tensor<1x1xf32> isSplat, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1xf16> = dense<0.3> : tensor<1x1xf32> isSplat, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf16>, tensor<1x1xf16> -> tensor<1x1x256xf16>

    %2 = IE.Slice %arg0 [0, 0, 256] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    %3 = IE.Multiply(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256xf16>, tensor<1x1xf16> -> tensor<1x1x256xf16>

    return %1, %3: tensor<1x1x256xf16>, tensor<1x1x256xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e-01> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e-01> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 2, 256]} : tensor<1x1x512xf16> -> tensor<1x1x2x256xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x1xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[LHS]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x256xf16>, tensor<1x1x2x1xf16> -> tensor<1x1x2x256xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 1, 512]} : tensor<1x1x2x256xf16> -> tensor<1x1x512xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 256] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0] [1, 1, 256] : tensor<1x1x512xf16> to tensor<1x1x256xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x1x256xf16>, tensor<1x1x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeParallelMultiplyLayersNHWC
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x32x64xf16, {order = #NHWC}>
func.func @MergeParallelMultiplyLayersNHWC(%arg0: tensor<1x16x32x64xf16, {order = #NHWC}>) -> (tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<0.2> : tensor<1x1x1x1xf16, {order = #NHWC}>
    %cst_1 = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<0.3> : tensor<1x1x1x1xf16, {order = #NHWC}>

    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 32, 32] : tensor<1x16x32x64xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>
    %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    %2 = IE.Slice %arg0 [0, 0, 0, 32] [1, 16, 32, 32] : tensor<1x16x32x64xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>
    %3 = IE.Multiply(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x16x32x32xf16, {order = #NHWC}>

    return %1, %3: tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<1.999510e-01> : tensor<1x1x1x1xf16, {order = #NHWC}>, [#const.MemPermute<#NCHW, #NCHW>, #const.Reshape<[1, 1, 1, 1, 1]>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<3.000490e-01> : tensor<1x1x1x1xf16, {order = #NHWC}>, [#const.MemPermute<#NCHW, #NCHW>, #const.Reshape<[1, 1, 1, 1, 1]>]

    // CHECK:       [[LHS_PERM_CAST:%.+]] = IE.PermuteCast([[INPUT]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x32x64xf16, {order = #NHWC}> -> tensor<1x32x64x16xf16>
    // CHECK:       [[LHS_RESHAPE:%.+]] = IE.Reshape([[LHS_PERM_CAST]]) {shape_value = [1, 32, 2, 32, 16]} : tensor<1x32x64x16xf16> -> tensor<1x32x2x32x16xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x2x1x1xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[LHS_RESHAPE]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x2x32x16xf16>, tensor<1x1x2x1x1xf16> -> tensor<1x32x2x32x16xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 32, 64, 16]} : tensor<1x32x2x32x16xf16> -> tensor<1x32x64x16xf16>
    // CHECK:       [[OUT_PERM_CAST:%.+]] = IE.PermuteCast([[OUT_RESHAPE]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x64x16xf16> -> tensor<1x16x32x64xf16, {order = #NHWC}>

    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[OUT_PERM_CAST]] [0, 0, 0, 32] [1, 16, 32, 32] : tensor<1x16x32x64xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[OUT_PERM_CAST]] [0, 0, 0, 0] [1, 16, 32, 32] : tensor<1x16x32x64xf16, {order = #NHWC}> to tensor<1x16x32x32xf16, {order = #NHWC}>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x16x32x32xf16, {order = #NHWC}>, tensor<1x16x32x32xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @MergeParallelMultiplyLayersSlicedOnInnerNonTrivialDim
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1536x2xf16>
func.func @MergeParallelMultiplyLayersSlicedOnInnerNonTrivialDim(%arg0: tensor<1x1536x2xf16>) -> (tensor<1x1536x1xf16>, tensor<1x1536x1xf16>) {
    %cst_0 = const.Declare tensor<1x1x1xf16> = dense<0.2> : tensor<1x1x1xf32> isSplat, [#const.CastElemType<f16>]
    %cst_1 = const.Declare tensor<1x1x1xf16> = dense<0.3> : tensor<1x1x1xf32> isSplat, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    %1 = IE.Multiply(%0, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x1xf16>, tensor<1x1x1xf16> -> tensor<1x1536x1xf16>

    %2 = IE.Slice %arg0 [0, 0, 1] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    %3 = IE.Multiply(%2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x1xf16>, tensor<1x1x1xf16> -> tensor<1x1536x1xf16>

    return %1, %3: tensor<1x1536x1xf16>, tensor<1x1536x1xf16>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e-01> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e-01> : tensor<1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:       [[LHS:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1536, 2, 1]} : tensor<1x1536x2xf16> -> tensor<1x1536x2x1xf16>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[CST]], [[CST_0]]) {per_axis = #IE.Concat<axis = 2 : i64>} : tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x2x1xf16>
    // CHECK:       [[MUL:%.+]] = IE.Multiply([[LHS]], [[CONCAT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x2x1xf16>, tensor<1x1x2x1xf16> -> tensor<1x1536x2x1xf16>
    // CHECK:       [[OUT_RESHAPE:%.+]] = IE.Reshape([[MUL]]) {shape_value = [1, 1536, 2]} : tensor<1x1536x2x1xf16> -> tensor<1x1536x2xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 1] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[OUT_RESHAPE]] [0, 0, 0] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x1536x1xf16>, tensor<1x1536x1xf16>
}

// -----

// CHECK-LABEL: @MergeParallelReshapeLayersWithDroppedDims
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1x512xf32>
func.func @MergeParallelReshapeLayersWithDroppedDims(%arg0: tensor<1x1x512xf32>) -> (tensor<1x256xf32>, tensor<1x256xf32>) {
    %0 = IE.Slice %arg0 [0, 0, 0] [1, 1, 256] : tensor<1x1x512xf32> to tensor<1x1x256xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>

    %2 = IE.Slice %arg0 [0, 0, 256] [1, 1, 256] : tensor<1x1x512xf32> to tensor<1x1x256xf32>
    %3 = IE.Reshape(%2) {shape_value = [1, 256]} : tensor<1x1x256xf32> -> tensor<1x256xf32>

    return %1, %3: tensor<1x256xf32>, tensor<1x256xf32>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]] [0, 256] [1, 256] : tensor<1x512xf32> to tensor<1x256xf32>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]] [0, 0] [1, 256] : tensor<1x512xf32> to tensor<1x256xf32>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x256xf32>, tensor<1x256xf32>
}

// -----

// CHECK-LABEL: @MergeParallelReshapeLayersWithPaddingDims
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x512xf32>
func.func @MergeParallelReshapeLayersWithPaddingDims(%arg0: tensor<1x512xf32>) -> (tensor<1x1x256xf32>, tensor<1x1x256xf32>) {
    %0 = IE.Slice %arg0 [0, 0] [1, 256] : tensor<1x512xf32> to tensor<1x256xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>

    %2 = IE.Slice %arg0 [0, 256] [1, 256] : tensor<1x512xf32> to tensor<1x256xf32>
    %3 = IE.Reshape(%2) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>

    return %1, %3: tensor<1x1x256xf32>, tensor<1x1x256xf32>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]] [1, 0, 256] [1, 1, 256] : tensor<1x1x512xf32> to tensor<1x1x256xf32>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]] [1, 0, 0] [1, 1, 256] : tensor<1x1x512xf32> to tensor<1x1x256xf32>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x1x256xf32>, tensor<1x1x256xf32>
}


// -----

// CHECK-LABEL: @MergeParallelReshapeLayersWithDroppedDimsSliceDimSizeToOne
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x1x2x640x32xf32>
func.func @MergeParallelReshapeLayersWithDroppedDimsSliceDimSizeToOne(%arg0: tensor<2x1x2x640x32xf32>) -> (tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<1x2x640x32xf32>

    %2 = IE.Slice %arg0 [1, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    %3 = IE.Reshape(%2) {shape_value = [1, 2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<1x2x640x32xf32>

    return %1, %3: tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [2, 2, 640, 32]} : tensor<2x1x2x640x32xf32> -> tensor<2x2x640x32xf32>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]] [1, 0, 0, 0] [1, 2, 640, 32] : tensor<2x2x640x32xf32> to tensor<1x2x640x32xf32>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [1, 2, 640, 32] : tensor<2x2x640x32xf32> to tensor<1x2x640x32xf32>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>
}

// -----

// CHECK-LABEL: @NotMergeParallelReshapeLayersWhenNoEnoughDimOfSizeOneToStrip
// CHECK-SAME:      [[INPUT:%.+]]: tensor<2x1x2x640x32xf32>
func.func @NotMergeParallelReshapeLayersWhenNoEnoughDimOfSizeOneToStrip(%arg0: tensor<2x1x2x640x32xf32>) -> (tensor<2x640x32xf32>, tensor<2x640x32xf32>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    %1 = IE.Reshape(%0) {shape_value = [2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<2x640x32xf32>

    %2 = IE.Slice %arg0 [1, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    %3 = IE.Reshape(%2) {shape_value = [2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<2x640x32xf32>

    return %1, %3: tensor<2x640x32xf32>, tensor<2x640x32xf32>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    // CHECK:       [[RESHAPE_0:%.+]] = IE.Reshape([[SLICE_0]]) {shape_value = [2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<2x640x32xf32>

    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT]] [1, 0, 0, 0, 0] [1, 1, 2, 640, 32] : tensor<2x1x2x640x32xf32> to tensor<1x1x2x640x32xf32>
    // CHECK:       [[RESHAPE_1:%.+]] = IE.Reshape([[SLICE_1]]) {shape_value = [2, 640, 32]} : tensor<1x1x2x640x32xf32> -> tensor<2x640x32xf32>

    // CHECK:       return [[RESHAPE_0]], [[RESHAPE_1]] : tensor<2x640x32xf32>, tensor<2x640x32xf32>
}

// -----

// CHECK-LABEL: @MergeParallelReshapeLayersWithDroppedMultipleNonAdjacentDims
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x2x1x2x640x32xf32>
func.func @MergeParallelReshapeLayersWithDroppedMultipleNonAdjacentDims(%arg0: tensor<1x2x1x2x640x32xf32>) -> (tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>) {
    %0 = IE.Slice %arg0 [0, 0, 0, 0, 0, 0] [1, 1, 1, 2, 640, 32] : tensor<1x2x1x2x640x32xf32> to tensor<1x1x1x2x640x32xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 2, 640, 32]} : tensor<1x1x1x2x640x32xf32> -> tensor<1x2x640x32xf32>

    %2 = IE.Slice %arg0 [0, 1, 0, 0, 0, 0] [1, 1, 1, 2, 640, 32] : tensor<1x2x1x2x640x32xf32> to tensor<1x1x1x2x640x32xf32>
    %3 = IE.Reshape(%2) {shape_value = [1, 2, 640, 32]} : tensor<1x1x1x2x640x32xf32> -> tensor<1x2x640x32xf32>

    return %1, %3: tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [2, 2, 640, 32]} : tensor<1x2x1x2x640x32xf32> -> tensor<2x2x640x32xf32>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[RESHAPE]] [1, 0, 0, 0] [1, 2, 640, 32] : tensor<2x2x640x32xf32> to tensor<1x2x640x32xf32>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[RESHAPE]] [0, 0, 0, 0] [1, 2, 640, 32] : tensor<2x2x640x32xf32> to tensor<1x2x640x32xf32>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x2x640x32xf32>, tensor<1x2x640x32xf32>
}

// -----

// CHECK-LABEL: @MergeParallelFullyConnectedLayersSliceWeightsOnD1
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1536x256xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x512xf16>
func.func @MergeParallelFullyConnectedLayersSliceWeightsOnD1(%arg0: tensor<1536x256xf16>, %arg1: tensor<1x512xf16>) -> (tensor<1536x1xf16>, tensor<1536x1xf16>) {
    %0 = IE.Slice %arg1 [0, 0] [1, 256] : tensor<1x512xf16> to tensor<1x256xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<1536x256xf16>, tensor<1x256xf16> -> tensor<1536x1xf16>

    %2 = IE.Slice %arg1 [0, 256] [1, 256] : tensor<1x512xf16> to tensor<1x256xf16>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1536x256xf16>, tensor<1x256xf16> -> tensor<1536x1xf16>

    return %1, %3: tensor<1536x1xf16>, tensor<1536x1xf16>

    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[INPUT_1]]) {shape_value = [2, 256]} : tensor<1x512xf16> -> tensor<2x256xf16>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT_0]], [[RESHAPE]]) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[FC]] [0, 1] [1536, 1] : tensor<1536x2xf16> to tensor<1536x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[FC]] [0, 0] [1536, 1] : tensor<1536x2xf16> to tensor<1536x1xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1536x1xf16>, tensor<1536x1xf16>
}

// -----

// CHECK-LABEL: @DoNotMergeParallelFullyConnectedLayersDueToSliceOnInnerDim
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1536x256xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<2x512xf16>
func.func @DoNotMergeParallelFullyConnectedLayersDueToSliceOnInnerDim(%arg0: tensor<1536x256xf16>, %arg1: tensor<2x512xf16>) -> (tensor<1536x2xf16>, tensor<1536x2xf16>) {
    %0 = IE.Slice %arg1 [0, 0] [2, 256] : tensor<2x512xf16> to tensor<2x256xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>

    %2 = IE.Slice %arg1 [0, 256] [2, 256] : tensor<2x512xf16> to tensor<2x256xf16>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>

    return %1, %3: tensor<1536x2xf16>, tensor<1536x2xf16>

    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[INPUT_1]] [0, 0] [2, 256] : tensor<2x512xf16> to tensor<2x256xf16>
    // CHECK:       [[FC_0:%.+]] = IE.FullyConnected([[INPUT_0]], [[SLICE_0]]) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[INPUT_1]] [0, 256] [2, 256] : tensor<2x512xf16> to tensor<2x256xf16>
    // CHECK:       [[FC_1:%.+]] = IE.FullyConnected([[INPUT_0]], [[SLICE_1]]) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>

    // CHECK:       return [[FC_0]], [[FC_1]] : tensor<1536x2xf16>, tensor<1536x2xf16>
}

// -----

// CHECK-LABEL: @MergeParallelFullyConnectedLayersSliceWeightsOnD0
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1536x256xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4x256xf16>
func.func @MergeParallelFullyConnectedLayersSliceWeightsOnD0(%arg0: tensor<1536x256xf16>, %arg1: tensor<4x256xf16>) -> (tensor<1536x2xf16>, tensor<1536x2xf16>) {
    %0 = IE.Slice %arg1 [0, 0] [2, 256] : tensor<4x256xf16> to tensor<2x256xf16>
    %1 = IE.FullyConnected(%arg0, %0) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>

    %2 = IE.Slice %arg1 [2, 0] [2, 256] : tensor<4x256xf16> to tensor<2x256xf16>
    %3 = IE.FullyConnected(%arg0, %2) : tensor<1536x256xf16>, tensor<2x256xf16> -> tensor<1536x2xf16>

    return %1, %3: tensor<1536x2xf16>, tensor<1536x2xf16>

    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[INPUT_0]], [[INPUT_1]]) : tensor<1536x256xf16>, tensor<4x256xf16> -> tensor<1536x4xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[FC]] [0, 2] [1536, 2] : tensor<1536x4xf16> to tensor<1536x2xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[FC]] [0, 0] [1536, 2] : tensor<1536x4xf16> to tensor<1536x2xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1536x2xf16>, tensor<1536x2xf16>
}

// -----

// CHECK-LABEL: @MergeParallelTanhLayers
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x1536x2xf16>
func.func @MergeParallelTanhLayers(%arg0: tensor<1x1536x2xf16>) -> (tensor<1x1536x1xf16>, tensor<1x1536x1xf16>) {
    %0 = IE.Slice %arg0 [0, 0, 0] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    %1 = IE.Tanh(%0) : tensor<1x1536x1xf16> -> tensor<1x1536x1xf16>

    %2 = IE.Slice %arg0 [0, 0, 1] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    %3 = IE.Tanh(%2) : tensor<1x1536x1xf16> -> tensor<1x1536x1xf16>

    return %1, %3: tensor<1x1536x1xf16>, tensor<1x1536x1xf16>

    // CHECK:       [[TANH:%.+]] = IE.Tanh([[INPUT]]) : tensor<1x1536x2xf16> -> tensor<1x1536x2xf16>
    // CHECK:       [[SLICE_1:%.+]] = IE.Slice [[TANH]] [0, 0, 1] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>
    // CHECK:       [[SLICE_0:%.+]] = IE.Slice [[TANH]] [0, 0, 0] [1, 1536, 1] : tensor<1x1536x2xf16> to tensor<1x1536x1xf16>

    // CHECK:       return [[SLICE_0]], [[SLICE_1]] : tensor<1x1536x1xf16>, tensor<1x1536x1xf16>
}

// -----

// CHECK-LABEL: @MergeParallelMulAddLayers
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1620x9x9x2xf16>
func.func @MergeParallelMulAddLayers(%arg0: tensor<1620x9x9x2xf16>) -> (tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>) {
    %cst1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst2 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst3 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst4 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %1 = IE.Multiply(%0, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %3 = IE.Multiply(%2, %cst2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %4 = IE.Add(%1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %5 = IE.Add(%3, %cst4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    return %4, %5 : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>

    // CHECK:    [[CST4:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST3:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT1:%.+]] = IE.Concat([[CST1]], [[CST2]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPED_INPUT]], [[CONCAT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE1:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[RESHAPE2:%.+]] = IE.Reshape([[RESHAPE1]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT2:%.+]] = IE.Concat([[CST3]], [[CST4]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE2]], [[CONCAT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE3:%.+]] = IE.Reshape([[ADD]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>

    // CHECK:    return [[SLICE0]], [[SLICE1]] : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>
}

// -----

// CHECK-LABEL: @MergeParallelMulAddLayersWithUnsortedOrder
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1620x9x9x2xf16>
func.func @MergeParallelMulAddLayersWithUnsortedOrder(%arg0: tensor<1620x9x9x2xf16>) -> (tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>) {
    %cst1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst2 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst3 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst4 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %1 = IE.Multiply(%0, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %3 = IE.Multiply(%2, %cst2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %4 = IE.Add(%1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %5 = IE.Add(%3, %cst4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    return %4, %5 : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>

    // CHECK:    [[CST3:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST4:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT1:%.+]] = IE.Concat([[CST2]], [[CST1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPED_INPUT]], [[CONCAT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE1:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[RESHAPE2:%.+]] = IE.Reshape([[RESHAPE1]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT2:%.+]] = IE.Concat([[CST4]], [[CST3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE2]], [[CONCAT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE3:%.+]] = IE.Reshape([[ADD]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>

    // CHECK:    return [[SLICE1]], [[SLICE0]] : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>
}

// -----

// CHECK-LABEL: @MergeParallelMulAddLayersWithDiffInputIdx
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1620x9x9x2xf16>
func.func @MergeParallelMulAddLayersWithDiffInputIdx(%arg0: tensor<1620x9x9x2xf16>) -> (tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>) {
    %cst1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst2 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst3 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst4 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %1 = IE.Multiply(%cst1, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1620x9x9x1xf16> -> tensor<1620x9x9x1xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    %3 = IE.Multiply(%2, %cst2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %4 = IE.Add(%1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %5 = IE.Add(%3, %cst4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    return %4, %5 : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>

    // CHECK:    [[CST3:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST4:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Reshape<[1, 1, 1, 1, 1]>, #const.CastElemType<f16>]

    // CHECK:    [[RESHAPED_INPUT:%.+]] = IE.Reshape([[INPUT]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT1:%.+]] = IE.Concat([[CST2]], [[CST1]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[MULTIPLY:%.+]] = IE.Multiply([[RESHAPED_INPUT]], [[CONCAT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE1:%.+]] = IE.Reshape([[MULTIPLY]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[RESHAPE2:%.+]] = IE.Reshape([[RESHAPE1]]) {shape_value = [1620, 9, 9, 2, 1]} : tensor<1620x9x9x2xf16> -> tensor<1620x9x9x2x1xf16>
    // CHECK:    [[CONCAT2:%.+]] = IE.Concat([[CST4]], [[CST3]]) {per_axis = #IE.Concat<axis = 3 : i64>} : tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x1x1x2x1xf16>

    // CHECK:    [[ADD:%.+]] = IE.Add([[RESHAPE2]], [[CONCAT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2x1xf16>, tensor<1x1x1x2x1xf16> -> tensor<1620x9x9x2x1xf16>

    // CHECK:    [[RESHAPE3:%.+]] = IE.Reshape([[ADD]]) {shape_value = [1620, 9, 9, 2]} : tensor<1620x9x9x2x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 1] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>
    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[RESHAPE3]] [0, 0, 0, 0] [1620, 9, 9, 1] : tensor<1620x9x9x2xf16> to tensor<1620x9x9x1xf16>

    // CHECK:    return [[SLICE1]], [[SLICE0]] : tensor<1620x9x9x1xf16>, tensor<1620x9x9x1xf16>
}

// -----

// CHECK-LABEL: @NotMergeParallelMulAddLayersWithOverlap
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1620x9x9x3xf16>
func.func @NotMergeParallelMulAddLayersWithOverlap(%arg0: tensor<1620x9x9x3xf16>) -> (tensor<1620x9x9x2xf16>, tensor<1620x9x9x2xf16>) {
    %cst1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst2 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst3 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst4 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    %1 = IE.Multiply(%0, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 1] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    %3 = IE.Multiply(%2, %cst2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    %4 = IE.Add(%1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    %5 = IE.Add(%3, %cst4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>

    return %4, %5 : tensor<1620x9x9x2xf16>, tensor<1620x9x9x2xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    // CHECK:    [[MULTIPLY0:%.+]] = IE.Multiply([[SLICE0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>

    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 1] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    // CHECK:    [[MULTIPLY1:%.+]] = IE.Multiply([[SLICE1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>

    // CHECK:    [[ADD0:%.+]] = IE.Add([[MULTIPLY0]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[ADD1:%.+]] = IE.Add([[MULTIPLY1]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>

    // CHECK:    return [[ADD0]], [[ADD1]] : tensor<1620x9x9x2xf16>, tensor<1620x9x9x2xf16>
}

// -----

// CHECK-LABEL: @NotMergeParallelMulAddLayersWithDiffSize
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1620x9x9x3xf16>
func.func @NotMergeParallelMulAddLayersWithDiffSize(%arg0: tensor<1620x9x9x3xf16>) -> (tensor<1620x9x9x2xf16>, tensor<1620x9x9x1xf16>) {
    %cst1 = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst2 = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst3 = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    %cst4 = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    %0 = IE.Slice %arg0 [0, 0, 0, 0] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    %1 = IE.Multiply(%0, %cst1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    %2 = IE.Slice %arg0 [0, 0, 0, 2] [1620, 9, 9, 1] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x1xf16>
    %3 = IE.Multiply(%2, %cst2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>
    %4 = IE.Add(%1, %cst3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    %5 = IE.Add(%3, %cst4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    return %4, %5 : tensor<1620x9x9x2xf16>, tensor<1620x9x9x1xf16>

    // CHECK:    [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:    [[CST2:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<4.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]

    // CHECK:    [[SLICE0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1620, 9, 9, 2] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x2xf16>
    // CHECK:    [[MULTIPLY0:%.+]] = IE.Multiply([[SLICE0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>

    // CHECK:    [[SLICE1:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 2] [1620,  9, 9, 1] : tensor<1620x9x9x3xf16> to tensor<1620x9x9x1xf16>
    // CHECK:    [[MULTIPLY1:%.+]] = IE.Multiply([[SLICE1]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    // CHECK:    [[ADD0:%.+]] = IE.Add([[MULTIPLY0]], [[CST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x2xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x2xf16>
    // CHECK:    [[ADD1:%.+]] = IE.Add([[MULTIPLY1]], [[CST2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1620x9x9x1xf16>, tensor<1x1x1x1xf16> -> tensor<1620x9x9x1xf16>

    // CHECK:    return [[ADD0]], [[ADD1]] : tensor<1620x9x9x2xf16>, tensor<1620x9x9x1xf16>
}

// -----

// CHECK-LABEL: @MoveParallelReshapeOpsWithPaddingDimsAfterConcat
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x256xf32>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x256xf32>
func.func @MoveParallelReshapeOpsWithPaddingDimsAfterConcat(%arg0: tensor<1x256xf32>, %arg1: tensor<1x256xf32>) -> tensor<1x1x512xf32> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>
    %1 = IE.Reshape(%arg1) {shape_value = [1, 1, 256]} : tensor<1x256xf32> -> tensor<1x1x256xf32>

    %2 = IE.Concat(%0, %1) {static_offsets = [[0, 0, 0], [0, 0, 256]]} : tensor<1x1x256xf32>, tensor<1x1x256xf32> -> tensor<1x1x512xf32>

    return %2: tensor<1x1x512xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x512xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>

    // CHECK:       return [[RESHAPE]] : tensor<1x1x512xf32>
}

// -----

// CHECK-LABEL: @MoveParallelReshapeOpsWithDroppedDimsAfterConcat
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1537xf32>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1537xf32>
func.func @MoveParallelReshapeOpsWithDroppedDimsAfterConcat(%arg0: tensor<1x1x1537xf32>, %arg1: tensor<1x1x1537xf32>) -> tensor<2x1537xf32> {
    %0 = IE.Reshape(%arg0) {shape_value = [1, 1537]} : tensor<1x1x1537xf32> -> tensor<1x1537xf32>
    %1 = IE.Reshape(%arg1) {shape_value = [1, 1537]} : tensor<1x1x1537xf32> -> tensor<1x1537xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1537xf32>, tensor<1x1537xf32> -> tensor<2x1537xf32>

    return %2: tensor<2x1537xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[CONCAT]]) {shape_value = [2, 1537]} : tensor<1x2x1537xf32> -> tensor<2x1537xf32>

    // CHECK:       return [[RESHAPE]] : tensor<2x1537xf32>
}

// -----

// CHECK-LABEL: @MoveParallelFullyConnectedAndSoftMaxLayersAfterConcat
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x1x1536xf32>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1536xf32>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<256x1536xf32>
func.func @MoveParallelFullyConnectedAndSoftMaxLayersAfterConcat(%arg0: tensor<1x1x1536xf32>, %arg1: tensor<1x1x1536xf32>, %arg2: tensor<256x1536xf32>) -> tensor<1x512xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
    %1 = IE.Reshape(%0) {shape_value = [1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1536xf32>
    %2 = IE.FullyConnected(%1, %arg2) : tensor<1x1536xf32>, tensor<256x1536xf32> -> tensor<1x256xf32>

    %3 = IE.SoftMax(%arg1) {axisInd = 2 : i64} : tensor<1x1x1536xf32> -> tensor<1x1x1536xf32>
    %4 = IE.Reshape(%3) {shape_value = [1, 1536]} : tensor<1x1x1536xf32> -> tensor<1x1536xf32>
    %5 = IE.FullyConnected(%4, %arg2) : tensor<1x1536xf32>, tensor<256x1536xf32> -> tensor<1x256xf32>

    %6 = IE.Concat(%2, %5) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x256xf32>, tensor<1x256xf32> -> tensor<1x512xf32>

    return %6: tensor<1x512xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1536xf32>, tensor<1x1x1536xf32> -> tensor<1x2x1536xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 2 : i64} : tensor<1x2x1536xf32> -> tensor<1x2x1536xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[SOFTMAX]]) {shape_value = [2, 1536]} : tensor<1x2x1536xf32> -> tensor<2x1536xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[RESHAPE]], [[INPUT_2]]) : tensor<2x1536xf32>, tensor<256x1536xf32> -> tensor<2x256xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 512]} : tensor<2x256xf32> -> tensor<1x512xf32>

    // CHECK:       return [[RESHAPE]] : tensor<1x512xf32>
}

// -----

// CHECK-LABEL: @MoveParallelFullyConnectedOpsAfterConcat
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x1537xf32>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1537xf32>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<2048x1537xf32>
func.func @MoveParallelFullyConnectedOpsAfterConcat(%arg0: tensor<1x1537xf32>, %arg1: tensor<1x1537xf32>, %arg2: tensor<2048x1537xf32>) -> tensor<1x4096xf32> {
    %0 = IE.FullyConnected(%arg0, %arg2) : tensor<1x1537xf32>, tensor<2048x1537xf32> -> tensor<1x2048xf32>
    %1 = IE.FullyConnected(%arg1, %arg2) : tensor<1x1537xf32>, tensor<2048x1537xf32> -> tensor<1x2048xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x2048xf32>, tensor<1x2048xf32> -> tensor<1x4096xf32>

    return %2: tensor<1x4096xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x1537xf32>, tensor<1x1537xf32> -> tensor<2x1537xf32>
    // CHECK:       [[FC:%.+]] = IE.FullyConnected([[CONCAT]], [[INPUT_2]]) : tensor<2x1537xf32>, tensor<2048x1537xf32> -> tensor<2x2048xf32>
    // CHECK:       [[RESHAPE:%.+]] = IE.Reshape([[FC]]) {shape_value = [1, 4096]} : tensor<2x2048xf32> -> tensor<1x4096xf32>

    // CHECK:       return [[RESHAPE]] : tensor<1x4096xf32>
}

// -----

// CHECK-LABEL: @NotMoveParallelFullyConnectedOpsAfterConcatDueToLargerInputSize
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1536x1536xf32>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1536x1536xf32>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<256x1536xf32>
func.func @NotMoveParallelFullyConnectedOpsAfterConcatDueToLargerInputSize(%arg0: tensor<1536x1536xf32>, %arg1: tensor<1536x1536xf32>, %arg2: tensor<256x1536xf32>) -> tensor<1536x512xf32> {
    %0 = IE.FullyConnected(%arg0, %arg2) : tensor<1536x1536xf32>, tensor<256x1536xf32> -> tensor<1536x256xf32>

    %1 = IE.FullyConnected(%arg1, %arg2) : tensor<1536x1536xf32>, tensor<256x1536xf32> -> tensor<1536x256xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1536x256xf32>, tensor<1536x256xf32> -> tensor<1536x512xf32>

    return %2: tensor<1536x512xf32>

    // CHECK:       [[FC_0:%.+]] = IE.FullyConnected([[INPUT_0]], [[INPUT_2]]) : tensor<1536x1536xf32>, tensor<256x1536xf32> -> tensor<1536x256xf32>
    // CHECK:       [[FC_1:%.+]] = IE.FullyConnected([[INPUT_1]], [[INPUT_2]]) : tensor<1536x1536xf32>, tensor<256x1536xf32> -> tensor<1536x256xf32>
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[FC_0]], [[FC_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1536x256xf32>, tensor<1536x256xf32> -> tensor<1536x512xf32>

    // CHECK:       return [[CONCAT]] : tensor<1536x512xf32>
}

// -----

// CHECK-LABEL: @MoveParallelSoftmaxOpsAfterConcat
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x1537xf32>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x1537xf32>
func.func @MoveParallelSoftmaxOpsAfterConcat(%arg0: tensor<1x1x1537xf32>, %arg1: tensor<1x1x1537xf32>) -> tensor<1x2x1537xf32> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x1537xf32> -> tensor<1x1x1537xf32>
    %1 = IE.SoftMax(%arg1) {axisInd = 2 : i64} : tensor<1x1x1537xf32> -> tensor<1x1x1537xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>

    return %2: tensor<1x2x1537xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>
    // CHECK:       [[SOFTMAX:%.+]] = IE.SoftMax([[CONCAT]]) {axisInd = 2 : i64} : tensor<1x2x1537xf32> -> tensor<1x2x1537xf32>

    // CHECK:       return [[SOFTMAX]] : tensor<1x2x1537xf32>
}

// -----

// CHECK-LABEL: @MoveParallelAddOpsAfterConcat
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x1x1537xf32>,
// CHECK-SAME:      [[INPUT_1:%arg[0-9]]]: tensor<1x1x1537xf32>,
// CHECK-SAME:      [[INPUT_2:%arg[0-9]]]: tensor<1x1537xf32>
func.func @MoveParallelAddOpsAfterConcat(%arg0: tensor<1x1x1537xf32>, %arg1: tensor<1x1x1537xf32>, %arg2: tensor<1x1537xf32>) -> tensor<1x2x1537xf32> {
    %0 = IE.Add(%arg0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1537xf32>, tensor<1x1537xf32> -> tensor<1x1x1537xf32>
    %1 = IE.Add(%arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1537xf32>, tensor<1x1537xf32> -> tensor<1x1x1537xf32>

    %2 = IE.Concat(%0, %1) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>

    return %2: tensor<1x2x1537xf32>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_0]], [[INPUT_1]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<1x1x1537xf32>, tensor<1x1x1537xf32> -> tensor<1x2x1537xf32>
    // CHECK:       [[ADD:%.+]] = IE.Add([[CONCAT]], [[INPUT_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1537xf32>, tensor<1x1537xf32> -> tensor<1x2x1537xf32>

    // CHECK:       return [[ADD]] : tensor<1x2x1537xf32>
}
