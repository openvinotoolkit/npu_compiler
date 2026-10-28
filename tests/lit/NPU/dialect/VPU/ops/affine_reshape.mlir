//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --canonicalize --platform=%platform% %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL: @Eliminate
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<4x4xf32>
func.func @Eliminate(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = VPU.AffineReshape(%arg0) { dim_mapping = [[0], [1]], shape_value = [4, 4] } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: VPU.AffineReshape
    // CHECK:     return [[ARG_0]]
}

// -----

// CHECK-LABEL: @ConstFold
func.func @ConstFold() -> tensor<4x4xf32> {
    %0 = const.Declare tensor<16xf32> = dense<1.0> : tensor<16xf32>
    %1 = VPU.AffineReshape(%0) { dim_mapping = [[0, 1]], shape_value = [4, 4] } : tensor<16xf32> -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>

    // CHECK-DAG:           [[VAL0:%.+]] = const.Declare tensor<4x4xf32> =
    // CHECK-SAME{LITERAL}: dense<1.000000e+00> : tensor<16xf32>, [#const.AffineReshape<[[0, 1]], [4, 4]>]
    // CHECK-NOT:   VPU.AffineReshape
    // CHECK:       return [[VAL0]]
}

// -----

func.func @SwapAffineReshapeSubView_Trivial() -> tensor<1x1x3xf32> {
    %cst = const.Declare tensor<1x2x3xf32> = dense<1.0> : tensor<1x2x3xf32>
    %affine_reshape = VPU.AffineReshape(%cst) {dim_mapping=[[0], [1], [2]], shape_value=[1, 2, 3]} : tensor<1x2x3xf32> -> tensor<1x2x3xf32>
    %slice = VPU.Slice %affine_reshape [0, 1, 0] [1, 1, 3] : tensor<1x2x3xf32> to tensor<1x1x3xf32>
    return %slice : tensor<1x1x3xf32>
    // CHECK-NOT: VPU.AffineReshape
    // CHECK-NOT: VPU.Slice
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1x1x3xf32> = dense<1.000000e+00> : tensor<1x2x3xf32>, [#const.SubView<[0, 1, 0], [1, 1, 3]>]
    // CHECK:     return [[CST]]
}

// -----

#HWC = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

func.func @SwapAffineReshapeAndSubView_Transpose() -> tensor<2x1x2xf32, {order=#HWC}> {
    %cst = const.Declare tensor<2x3x4xf32> = dense<1.0> : tensor<2x3x4xf32>
    // This AffineReshape is just a simple transpose and therefore we can order SubView before.
    %affine_reshape = VPU.AffineReshape(%cst) {dim_mapping=[[1], [2], [0]], shape_value=[4, 2, 3]} : tensor<2x3x4xf32> -> tensor<4x2x3xf32, {order=#HWC}>
    %slice = VPU.Slice %affine_reshape [2, 1, 0] [2, 1, 2] : tensor<4x2x3xf32, {order=#HWC}> to tensor<2x1x2xf32, {order=#HWC}>
    return %slice : tensor<2x1x2xf32, {order=#HWC}>
    // CHECK-NOT: VPU.AffineReshape
    // CHECK-NOT: VPU.Slice
    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x1x2xf32, {order = #HWC}> = dense<1.000000e+00> : tensor<2x3x4xf32>
    // CHECK-SAME{LITERAL}:     [#const.SubView<[1, 0, 2], [1, 2, 2]>, #const.AffineReshape<[[1], [2], [0]], [2, 1, 2]>]
    // CHECK:     return [[CST]] : tensor<2x1x2xf32, {order = #HWC}>
}

// -----

//  [(0, 0)*, (0, 1)*, (0, 2)]
//  [(1, 0)*, (1, 1)*, (1, 2)]
//  [(2, 0),  (2, 1),  (2, 2)]
//  [(3, 0),  (3, 1),  (3, 2)]
// Legal: Maps to
//  [(0, 0, 0), (0, 0, 1)]
//  [(0, 1, 0), (0, 1, 1)]
// in the input tensor.
func.func @SwapAffineReshapeAndSubView() -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2x2x3xf32> = dense<1.0> : tensor<2x2x3xf32>
    %affine_reshape = VPU.AffineReshape(%cst) {dim_mapping=[[0], [0], [1]], shape_value=[4, 3]} : tensor<2x2x3xf32> -> tensor<4x3xf32>
    %slice = VPU.Slice %affine_reshape [0, 0] [2, 2] : tensor<4x3xf32> to tensor<2x2xf32>
    return %slice : tensor<2x2xf32>
    // CHECK-NOT: VPU.AffineReshape
    // CHECK-NOT: VPU.Slice
    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2x3xf32>
    // CHECK-SAME{LITERAL}:     [#const.SubView<[0, 0, 0], [1, 2, 2]>, #const.AffineReshape<[[0], [0], [1]], [2, 2]>]
    // CHECK:     return [[CST]]
}

// -----

// Note: The following test cases describe different subviews when reshaping 2x2x3 to 4x3. We mark the elements that are
// selected by subview with (*).
//       [(0, 0),  (0, 1),  (0, 2)]
//       [(1, 0)*, (1, 1)*, (1, 2)]
//       [(2, 0)*, (2, 1)*, (2, 2)]
//       [(3, 0),  (3, 1),  (3, 2)]
// Illegal: Maps to
//       [(0, 1, 0), (0, 1, 1)]
//       [(1, 0, 0), (1, 0, 1)]
// in the input tensor.
func.func @DoNotSwapAffineReshapeAndSubView() -> tensor<2x2xf32> {
    %cst = const.Declare tensor<2x2x3xf32> = dense<1.0> : tensor<2x2x3xf32>
    %affine_reshape = VPU.AffineReshape(%cst) {dim_mapping=[[0], [0], [1]], shape_value=[4, 3]} : tensor<2x2x3xf32> -> tensor<4x3xf32>
    %slice = VPU.Slice %affine_reshape [1, 0] [2, 2] : tensor<4x3xf32> to tensor<2x2xf32>
    return %slice : tensor<2x2xf32>
    // CHECK-NOT: VPU.AffineReshape
    // CHECK-NOT: VPU.Slice
    // CHECK:     [[CST:%.+]] = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2x3xf32>
    // CHECK-SAME{LITERAL}:     [#const.AffineReshape<[[0], [0], [1]], [4, 3]>, #const.SubView<[1, 0], [2, 2]>]
    // CHECK:     return [[CST]]
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!Segmented5D = !VPU.DistributedTensor<
    2x1x48x1x1xf16, #NCDHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1, 1],
    num_clusters = 3 : i64,
    alignment = [1, 1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[2, 1, 16, 1, 1], [2, 1, 16, 1, 1], [2, 1, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 32, 0, 0]],
    memory_shapes = [[2, 1, 16, 1, 1], [2, 1, 16, 1, 1], [2, 1, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [0, 0, 16, 0, 0], [0, 0, 32, 0, 0]]
}>

!Segmented4D = !VPU.DistributedTensor<
    2x48x1x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[2, 16, 1, 1], [2, 16, 1, 1], [2, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[2, 16, 1, 1], [2, 16, 1, 1], [2, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
}>

// CHECK-LABEL: @RankChangingSegmentedDistribution
func.func @RankChangingSegmentedDistribution(%arg0: !Segmented5D) -> !Segmented4D {
    // CHECK: VPU.AffineReshape
    // CHECK-SAME: num_tiles = [1, 3, 1, 1]
    // CHECK-SAME: alignment = [1, 16, 1, 1]
    %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [2, 48, 1, 1]} : !Segmented5D -> !Segmented4D
    return %0 : !Segmented4D
}
