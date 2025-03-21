//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

// CHECK-LABEL: @OneInputFold
func.func @OneInputFold(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = VPU.Concat(%arg0) { per_axis = #IE.Concat<axis = 1> } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: VPU.Concat
    // CHECK:     return %arg0
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseRightConcat
func.func @FuseRightConcat(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                           %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                           %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %TILING = VPU.Concat(%arg1, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 64, 0, 0],
    // CHECK-SAME:           [0, 64, 125, 0]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x64x250x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseLeftConcat
func.func @FuseLeftConcat(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg2: tensor<1x64x250x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%TILING, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 0, 125, 0],
    // CHECK-SAME:           [0, 32, 0, 0]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x250xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseTwoConcats
func.func @FuseTwoConcats(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg2: tensor<1x64x250x125xf16, {order = #NHWC}>,
                          %arg3: tensor<1x64x250x125xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %LHS_TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %RHS_TILING = VPU.Concat(%arg2, %arg3) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 0, 125]
        ]
    } : tensor<1x64x250x125xf16, {order = #NHWC}>,
        tensor<1x64x250x125xf16, {order = #NHWC}>
            -> tensor<1x64x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%LHS_TILING, %RHS_TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2, %arg3) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 0, 125, 0],
    // CHECK-SAME:           [0, 32, 0, 0]
    // CHECK-SAME:           [0, 32, 0, 125]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x125xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x125xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipConcatWithTwoConsumers
func.func @SkipConcatWithTwoConsumers(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                                      %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                                      %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x32x250x250xf16, {order = #NHWC}>)  {

    %TILING = VPU.Concat(%arg1, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT, %TILING :
        tensor<1x96x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>

    // CHECK:   [[TILING_CONCAT:%.*]] = VPU.Concat(%arg1, %arg2)
    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, [[TILING_CONCAT]])

    // CHECK:   return [[MAIN_CONCAT]], [[TILING_CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseConcatWithTwoConsumers
func.func @FuseConcatWithTwoConsumers(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                                      %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                                      %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x96x250x250xf16, {order = #NHWC}>) {

    %TILING = VPU.Concat(%arg1, %arg2) {
        per_axis = #IE.Concat<axis = 2>
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    %ANOTHER_CONCAT = VPU.Concat(%TILING, %arg0) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT, %ANOTHER_CONCAT :
                tensor<1x96x250x250xf16, {order = #NHWC}>,
                tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 64, 0, 0],
    // CHECK-SAME:          [0, 64, 125, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x64x250x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[ANOTHER_CONCAT:%.*]] = VPU.Concat(%arg1, %arg2, %arg0) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 125, 0],
    // CHECK-SAME:          [0, 32, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x64x250x250xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return %0, %1 : tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseConstants
func.func @FuseConstants(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %CST_PRODUCER = const.Declare tensor<1x64x250x250xf16, {order = #NHWC}> =
        dense<1.0> : tensor<1x64x250x250xf16>, [#const.Reorder<#NHWC>]

    %TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%CST_PRODUCER, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[CST_PRODUCER:%.*]] = const.Declare
    // CHECK:   [[CONCAT:%.*]] = VPU.Concat([[CST_PRODUCER]], %arg0, %arg1)

    // CHECK:   return [[CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPU.DistributedTensor<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 32, 65, 128], [1, 32, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

!InputDistributed1 = !VPU.DistributedTensor<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 16, 65, 128], [1, 16, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]],
    memory_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

// CHECK-LABEL: ConcatWithExplicitOverlappedDistributedTensorType
func.func @ConcatWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed0,
                                                             %arg1: !InputDistributed1)
    -> !OutputDistributed {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : !InputDistributed0, !InputDistributed1 -> !OutputDistributed

    return %concat : !OutputDistributed

    // CHECK:        [[CONCAT:%.*]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:         !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 65, 128], [1, 32, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]

    // CHECK-SAME:         !VPU.DistributedTensor<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 65, 128], [1, 16, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]

    // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPU.DistributedTensor<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

!InputDistributed1 = !VPU.DistributedTensor<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

// CHECK-LABEL: ConcatWithExplicitSegmentedDistributedTensorType
func.func @ConcatWithExplicitSegmentedDistributedTensorType(%arg0: !InputDistributed0,
                                                             %arg1: !InputDistributed1)
    -> !OutputDistributed {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : !InputDistributed0, !InputDistributed1 -> !OutputDistributed

    return %concat : !OutputDistributed

    // CHECK:        [[CONCAT:%.*]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:         !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]

    // CHECK-SAME:         !VPU.DistributedTensor<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]

    // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipFuseConcatWithSwOp
// CHECK-SAME: ([[INPUT0:%.*]]: tensor<1x128x16x32xf16, {order = #NHWC}>, [[INPUT1:%.*]]: tensor<1x128x16x32xf16, {order = #NHWC}>)
func.func @SkipFuseConcatWithSwOp(%arg0: tensor<1x128x16x32xf16, {order = #NHWC}>, %arg1: tensor<1x128x16x32xf16, {order = #NHWC}>)
    -> (tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>) {
    %cst0 = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]

    %gelu_in = VPU.Gelu(%arg0) : tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x32xf16, {order = #NHWC}>

    %main_concat = VPU.Concat(%gelu_in, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x128x16x32xf16, {order = #NHWC}>, tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>

    %concat0 = VPU.Concat(%main_concat, %cst0) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>
    %concat1 = VPU.Concat(%main_concat, %cst1) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %concat0 [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>
    %slice1 = VPU.Slice %concat1 [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>

    %gelu0 = VPU.Gelu(%slice0) : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %gelu1 = VPU.Gelu(%slice1) : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    return %gelu0, %gelu1 : tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>

    // CHECK:  [[CST0:%.*]] = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]
    // CHECK:  [[CST1:%.*]] = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]

    // CHECK:  [[GELU_INPUT:%.*]] = VPU.Gelu([[INPUT0]]) : tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x32xf16, {order = #NHWC}>
    // CHECK:  [[MAIN_CONCAT:%.*]] = VPU.Concat([[GELU_INPUT]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x128x16x32xf16, {order = #NHWC}>, tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:  [[CONCAT0:%.*]] = VPU.Concat([[MAIN_CONCAT]], [[CST0]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>
    // CHECK:  [[CONCAT1:%.*]] = VPU.Concat([[MAIN_CONCAT]], [[CST1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>

    // CHECK:  [[SLICE0:%.*]] = VPU.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>
    // CHECK:  [[SLICE1:%.*]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>

    // CHECK:  [[GELU0:%.*]] = VPU.Gelu([[SLICE0]]) : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    // CHECK:  [[GELU1:%.*]] = VPU.Gelu([[SLICE1]]) : tensor<1x256x16x32xf16, {order = #NHWC}> -> tensor<1x256x16x32xf16, {order = #NHWC}>
    // CHECK:  return [[GELU0]], [[GELU1]] : tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: SkipFuseConcatWithNCEOpNoWeights
// CHECK-SAME: ([[INPUT0:%.*]]: tensor<1x128x16x32xf16, {order = #NHWC}>, [[INPUT1:%.*]]: tensor<1x128x16x32xf16, {order = #NHWC}>)
func.func @SkipFuseConcatWithNCEOpNoWeights(%arg0: tensor<1x128x16x32xf16, {order = #NHWC}>, %arg1: tensor<1x128x16x32xf16, {order = #NHWC}>)
    -> (tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>) {
    %cst0 = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]
    %cst1 = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]

    %weights_table = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32>
    %maxpool = VPU.NCE.MaxPool(%arg0, %weights_table) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x128x16x32xf16, {order = #NHWC}>
    %main_concat = VPU.Concat(%maxpool, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x128x16x32xf16, {order = #NHWC}>, tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>

    %concat0 = VPU.Concat(%main_concat, %cst0) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>
    %concat1 = VPU.Concat(%main_concat, %cst1) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>

    %slice0 = VPU.Slice %concat0 [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>
    %slice1 = VPU.Slice %concat1 [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>

    %weights_table_0 = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>
    %weights_table_1 = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>
    %maxpool0 = VPU.NCE.MaxPool(%slice0, %weights_table_0) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } -> tensor<1x256x16x32xf16, {order = #NHWC}>
    %maxpool1 = VPU.NCE.MaxPool(%slice1, %weights_table_1) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    } -> tensor<1x256x16x32xf16, {order = #NHWC}>
    return %maxpool0, %maxpool1 : tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>

    // CHECK:  [[CST:%.*]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>
    // CHECK:  [[CST0:%.*]] = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]
    // CHECK:  [[CST1:%.*]] = const.Declare tensor<1x128x16x64xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x128x16x64xf16>, [#const.Reorder<#NHWC>]
    // CHECK:  [[CST2:%.*]] = const.Declare tensor<128x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32>

    // CHECK:  [[MAXPOOL_IN:%.*]] = VPU.NCE.MaxPool([[INPUT0]], [[CST2]] ) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x128x16x32xf16, {order = #NHWC}>
    // CHECK:  [[MAIN_CONCAT:%.*]] = VPU.Concat([[MAXPOOL_IN]], [[INPUT1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 32]]} : tensor<1x128x16x32xf16, {order = #NHWC}>, tensor<1x128x16x32xf16, {order = #NHWC}> -> tensor<1x128x16x64xf16, {order = #NHWC}>

    // CHECK:  [[CONCAT0:%.*]] = VPU.Concat([[MAIN_CONCAT]], [[CST0]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>
    // CHECK:  [[CONCAT1:%.*]] = VPU.Concat([[MAIN_CONCAT]], [[CST1]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x16x64xf16, {order = #NHWC}>, tensor<1x128x16x64xf16, {order = #NHWC}> -> tensor<1x256x16x64xf16, {order = #NHWC}>

    // CHECK:  [[SLICE0:%.*]] = VPU.Slice [[CONCAT0]] [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>
    // CHECK:  [[SLICE1:%.*]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 256, 16, 32] : tensor<1x256x16x64xf16, {order = #NHWC}> to tensor<1x256x16x32xf16, {order = #NHWC}>

    // CHECK:  [[MAXPOOL0:%.*]] = VPU.NCE.MaxPool([[SLICE0]], [[CST]] )
    // CHECK:  [[MAXPOOL1:%.*]] = VPU.NCE.MaxPool([[SLICE1]], [[CST]] )
    // CHECK:  return [[MAXPOOL0]], [[MAXPOOL1]] : tensor<1x256x16x32xf16, {order = #NHWC}>, tensor<1x256x16x32xf16, {order = #NHWC}>
}
