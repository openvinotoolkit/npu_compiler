//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x1x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x1x49x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}>

// CHECK-LABEL: SliceWithExplicitOverlappedDistributedTensorType
func.func @SliceWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    %1 = VPU.Slice %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SLICE0:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPU.DistributedTensor<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x1x49x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]

    // CHECK:        [[SLICE1:%.*]] = VPU.Slice %arg0 [0, 0, 47, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPU.DistributedTensor<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x1x49x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPU.DistributedTensor<
    1x16x88x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]],
    memory_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x16x44x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]],
    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
}>

// CHECK-LABEL: SliceWithExplicitSegmentedDistributedTensorType
func.func @SliceWithExplicitSegmentedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
    %1 = VPU.Slice %arg0 [0, 0, 44, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SLICE0:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPU.DistributedTensor<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x16x44x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

    // CHECK:        [[SLICE1:%.*]] = VPU.Slice %arg0 [0, 0, 44, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPU.DistributedTensor<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPU.DistributedTensor<1x16x44x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePattern
// CHECK-SAME:        ([[INPUT:%.+]]: tensor<1x16x1x1xf16>)
func.func @OptimizeExpandSlicePattern(%input: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
   %expand = VPU.Expand(%input) {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} : tensor<1x16x1x1xf16> -> tensor<16x16x1x1xf16>
   %slice = VPU.Slice %expand [0, 0, 0, 0] [1, 16, 1, 1] : tensor<16x16x1x1xf16> to tensor<1x16x1x1xf16>
   return %slice : tensor<1x16x1x1xf16>

   // CHECK-NOT:    VPU.Expand
   // CHECK-NOT:    VPU.Slice
   // CHECK:        return [[INPUT]] : tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @OptimizeExpandSlicePatternPadBothSides
// CHECK-SAME:        ([[INPUT:%.+]]: tensor<1x16x1x1xf16>)
func.func @OptimizeExpandSlicePatternPadBothSides(%input: tensor<1x16x1x1xf16>) -> tensor<1x16x1x1xf16> {
   %expand = VPU.Expand(%input) {pads_begin = [4, 0, 0, 0], pads_end = [15, 0, 0, 0]} : tensor<1x16x1x1xf16> -> tensor<20x16x1x1xf16>
   %slice = VPU.Slice %expand [4, 0, 0, 0] [1, 16, 1, 1] : tensor<20x16x1x1xf16> to tensor<1x16x1x1xf16>
   return %slice : tensor<1x16x1x1xf16>

   // CHECK-NOT:    VPU.Expand
   // CHECK-NOT:    VPU.Slice
   // CHECK:        return [[INPUT]] : tensor<1x16x1x1xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

!OutputDistributed = tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @SliceWithDynamicHWTensorType
// CHECK-SAME:        ([[INPUT:%.+]]: tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>)
func.func @SliceWithDynamicHWTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 12, -9223372036854775808, -9223372036854775808] : !InputDistributed to !OutputDistributed
    return %0 : !OutputDistributed
	
    // CHECK:        [[SLICE:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 12, -9223372036854775808, -9223372036854775808] : tensor<1x16x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:        return [[SLICE]] : tensor<1x12x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = tensor<1x16x800x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>

!OutputDistributed = tensor<1x12x400x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 400, 1280]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @SliceWithDynamicWTensorType
// CHECK-SAME:        ([[INPUT:%.+]]: tensor<1x16x800x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}>)
func.func @SliceWithDynamicWTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 12, 400, -9223372036854775808] : !InputDistributed to !OutputDistributed
    return %0 : !OutputDistributed
	
    // CHECK:        [[SLICE:%.*]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 12, 400, -9223372036854775808] : tensor<1x16x800x?xf32, {bounds = #const.OpaqueI64Elements<[1, 16, 800, 1280]> : tensor<4xsi64>, order = #NCHW}> to tensor<1x12x400x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 400, 1280]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:        return [[SLICE]] : tensor<1x12x400x?xf32, {bounds = #const.OpaqueI64Elements<[1, 12, 400, 1280]> : tensor<4xsi64>, order = #NCHW}>

}
