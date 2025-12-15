//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Fold
func.func @Fold(%arg0: memref<1x3x8x4xf32, #NHWC>) -> memref<1x3x8x4xf32, #NHWC> {
    %0 = const.Declare memref<1x3x16x16xf32, #NHWC> =
        dense<1.000000e+00> : tensor<1x3x16x16xf32>, [#const.Reorder<#NHWC>]

    %1 = VPUIP.SubView %0 [0, 0, 8, 12] [1, 3, 8, 4] :
        memref<1x3x16x16xf32, #NHWC> to
        memref<1x3x8x4xf32, {order = #NHWC, strides = [768, 1, 48, 3]}>

    %2 = VPUIP.Copy
        inputs(%1 : memref<1x3x8x4xf32, {order = #NHWC, strides = [768, 1, 48, 3]}>)
        outputs(%arg0 : memref<1x3x8x4xf32, #NHWC>)
        -> memref<1x3x8x4xf32, #NHWC>

    return %2 : memref<1x3x8x4xf32, #NHWC>

    // CHECK-DAG:       [[VAR0:%.+]] = const.Declare memref<1x3x8x4xf32, #NHWC> =
    // CHECK-SAME:      [#const.SubView<[0, 0, 8, 12], [1, 3, 8, 4]>, #const.Reorder<#NHWC>]
    // CHECK-NOT:   VPUIP.SubView

    // CHECK:       [[VAR1:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[VAR0]] : memref<1x3x8x4xf32, #NHWC>)
    // CHECK-SAME:      outputs(%arg0 : memref<1x3x8x4xf32, #NHWC>)

    // CHECK:       return [[VAR1]] : memref<1x3x8x4xf32, #NHWC>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @VPU.SW {
func.func private @builtin_relu(%input : memref<*xf16>, %output : memref<*xf16>) attributes {
        VPU.kernel_code = "activation_relu.cpp", VPU.kernel_entry = "activation_relu", VPU.task_type = @COMPUTE
    }
}

// CHECK-LABEL: @ComposeSubView
func.func @ComposeSubView(%arg0: memref<1x3x8x4xf32>) -> memref<1x3x8x4xf32> {
    %0 = memref.alloc() : memref<1x3x16x16xf32>

    %1 = VPUIP.SubView %0 [0, 0, 0, 8] [1, 3, 16, 8] :
        memref<1x3x16x16xf32> to
        memref<1x3x16x8xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    %2 = VPUIP.SubView %1 [0, 0, 8, 4] [1, 3, 8, 4] :
        memref<1x3x16x8xf32, {order = #NCHW, strides = [768, 256, 16, 1]}> to
        memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    %3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_relu
        inputs(%2 as %input: memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>)
        outputs(%arg0 as %output: memref<1x3x8x4xf32>) on tile 0 -> memref<1x3x8x4xf32> {
        VPUIP.SW.Kernel.run {attrs = [false, true, 6.0892105102539063E-4]} (%input, %output) : memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>, memref<1x3x8x4xf32>
    }

    return %3 : memref<1x3x8x4xf32>

    // CHECK:       [[VAR0:%.+]] = memref.alloc() : memref<1x3x16x16xf32>

    // CHECK:       [[VAR1:%.*]] = VPUIP.SubView [[VAR0]] [0, 0, 8, 12] [1, 3, 8, 4] :
    // CHECK-SAME:      memref<1x3x16x16xf32> to memref<1x3x8x4xf32, {order = #NCHW, strides = [768, 256, 16, 1]}>

    // CHECK:       [[VAR2:%.*]] = VPUIP.SW.Kernel
    // CHECK-SAME:      inputs([[VAR1]]
    // CHECK-SAME:      outputs(%arg0
    // CHECK-SAME:      -> memref<1x3x8x4xf32>

    // CHECK:       return [[VAR2]] : memref<1x3x8x4xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 1, 49, 160], [1, 1, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x1x49x160xf16, {order = #NHWC, strides = [15360, 1, 160, 1]}, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}>

// CHECK-LABEL: SubviewWithExplicitOverlappedDistributedTensorType
func.func @SubviewWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

    %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    %1 = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1x1x49x160xf16, {order = #NHWC, strides = [15360, 1, 160, 1]}, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]

    // CHECK:        [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 47, 0] [1, 1, 49, 160]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x1x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1x1x49x160xf16, {order = #NHWC, strides = [15360, 1, 160, 1]}, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 1, 26, 160], [1, 1, 25, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x88x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]],
    memory_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x44x128xf16, {order = #NHWC, strides = [180224, 1, 2048, 16]}, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]],
    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
}>

// CHECK-LABEL: SubviewWithExplicitSegmentedDistributedTensorType
func.func @SubviewWithExplicitSegmentedDistributedTensorType(%arg0: !InputDistributed)
    -> (!OutputDistributed, !OutputDistributed) {

     %0 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
     %1 = VPUIP.SubView %arg0 [0, 0, 44, 0] [1, 16, 44, 128] : !InputDistributed to !OutputDistributed
    return %0, %1 : !OutputDistributed, !OutputDistributed

    // CHECK:        [[SUBVIEW0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1x16x44x128xf16, {order = #NHWC, strides = [180224, 1, 2048, 16]}, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

    // CHECK:        [[SUBVIEW1:%.*]] = VPUIP.SubView %arg0 [0, 0, 44, 0] [1, 16, 44, 128]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1x16x44x128xf16, {order = #NHWC, strides = [180224, 1, 2048, 16]}, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 22, 128], [1, 16, 22, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 22, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x48x88x128xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    alignment = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 88, 128], [1, 16, 88, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 48, 88, 128], [1, 48, 88, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x3x88x128xf16, {order = #NHWC, strides = [540672, 1, 6144, 48]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 3, 88, 128], [1, 3, 88, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 3, 88, 128], [1, 3, 88, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: SubviewWithExplicitFullMemoryWithoutAlignDistributedTensorType
func.func @SubviewWithExplicitFullMemoryWithoutAlignDistributedTensorType(%arg0: !InputDistributed) -> !OutputDistributed {
     %0 = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 3, 88, 128] : !InputDistributed to !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SUBVIEW:%.*]] = VPUIP.SubView %arg0 [0, 2, 0, 0] [1, 3, 88, 128]
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x48x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         to !VPUIP.DistributedBuffer<1x3x88x128xf16, {order = #NHWC, strides = [540672, 1, 6144, 48]}, @CMX_NN
    // CHECK-SAME:             mode = "DUPLICATED|SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 3, 88, 128], [1, 3, 88, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 3, 88, 128], [1, 3, 88, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}
