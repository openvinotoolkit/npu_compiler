//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-subview-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

// CHECK: @OptimizeSubviewCopyConvPattern
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @OptimizeSubviewCopyConvPattern(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed){

    %subview = VPUIP.SubView %input [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc = VPURT.AllocDistributed -> !InputDistributed

    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv: !OutputDistributed

    // CHECK:       [[NEW_BUFFER:%.+]] =  VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[NEW_BUFFER]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @OptimizeSubviewCopy2ConvPattern
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @OptimizeSubviewCopy2ConvPattern(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %subview = VPUIP.SubView %input [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[NEW_BUFFER:%.+]] =  VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[NEW_BUFFER]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPattern
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @Optimize2SubviewCopyConvPattern(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %subview0 = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %input [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[NEW_BUFFER:%.+]] =  VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[NEW_BUFFER]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputDistributed = !VPUIP.DistributedBuffer<
    1x3072x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 1536, 0, 0]],
    memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternWithOptimizableCopyIn
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN
func.func @Optimize2SubviewCopyConvPatternWithOptimizableCopyIn(
        %input: !RootInputDistributed,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : !RootInputDistributed)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1


    // CHECK:        [[DISTRIB_CAST:%.+]] = VPUIP.DistributedCast
    // CHECK-SAME:      inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[DISTRIB_CAST]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[DISTRIB_CAST]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputDistributed = !VPUIP.DistributedBuffer<
    1x3072x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 1536, 0, 0]],
    memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternKeepCopyInWithOtherUsers
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN
func.func @Optimize2SubviewCopyConvPatternKeepCopyInWithOtherUsers(
        %input: !RootInputDistributed,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1, memref<1x3072x1x1xf16, #NHWC, @DDR>){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : !RootInputDistributed)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1, %spillCopy: !OutputDistributed0, !OutputDistributed1, memref<1x3072x1x1xf16, #NHWC, @DDR>

    // CHECK:        [[SPILL_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SPILL_COPY]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW0]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:          input([[SUBVIEW1]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                      {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                      {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternKeepCopyInNotCMX2DDR
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16,  {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @DDR>
func.func @Optimize2SubviewCopyConvPatternKeepCopyInNotCMX2DDR(
        %input: memref<1x3072x1x1xf16,  {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : memref<1x3072x1x1xf16,  {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @DDR>)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:        [[DDR2DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16,  {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[DDR2DDR]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputDistributed = !VPUIP.DistributedBuffer<
    1x3072x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1],
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 1536, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 1536, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternKeepCopyInNotDuplicatedLike
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN
func.func @Optimize2SubviewCopyConvPatternKeepCopyInNotDuplicatedLike(
        %input: !RootInputDistributed,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : !RootInputDistributed)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:        [[SEGMENTED_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1]
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SEGMENTED_COPY]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputDistributed = !VPUIP.DistributedBuffer<
    1x3072x1x1xf16, {order = #NHWC, strides = [50000000, 1, 50000000, 50000000]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternWithKeepCopyInNotFitInCMX
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [50000000, 1, 50000000, 50000000]}, @CMX_NN
func.func @Optimize2SubviewCopyConvPatternWithKeepCopyInNotFitInCMX(
        %input: !RootInputDistributed,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : !RootInputDistributed)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:        [[TO_DDR_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16,
    // CHECK-SAME:                             {order = #NHWC, strides = [50000000, 1, 50000000, 50000000]}, @CMX_NN,
    // CHECK-SAME:                                  mode = "DUPLICATED"
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:  -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    // CHECK:        [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[TO_DDR_COPY]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputDistributed = !VPUIP.DistributedBuffer<
    1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1536, 1, 1], [128, 1536, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    compute_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]],
    memory_shapes = [[128, 1, 1, 4], [128, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [128, 0, 0, 0]]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]],
    memory_shapes = [[1, 256, 1, 1], [1, 256, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK: @Optimize2SubviewCopyConvPatternWithOptimizeCopyInWithStrides
// CHECK-SAME: ([[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN
func.func @Optimize2SubviewCopyConvPatternWithOptimizeCopyInWithStrides(
        %input: !RootInputDistributed,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %ddrAlloc = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spillCopy = VPUIP.Copy inputs(%input : !RootInputDistributed)
                    outputs(%ddrAlloc : memref<1x3072x1x1xf16, #NHWC, @DDR>)
        -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spillCopy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %subview1 = VPUIP.SubView %spillCopy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN,
    // CHECK-SAME:           mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN,
    // CHECK-SAME:           mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ARG0]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN,
    // CHECK-SAME:           mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3072, 1, 1], [1, 3072, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN,
    // CHECK-SAME:           mode = "DUPLICATED", num_clusters = 2 : i64,
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1536, 1, 1], [1, 1536, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed0 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

!OutputDistributed1 = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @OptimizeSubview2CopyConvPattern
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @OptimizeSubview2CopyConvPattern(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed0, !OutputDistributed1){

    %subview = VPUIP.SubView %input [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    %alloc0 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy0 = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                                 outputs(%alloc0 : !InputDistributed)
        -> !InputDistributed

    %allocConv0 = VPURT.AllocDistributed -> !OutputDistributed0
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy0 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy0 : !InputDistributed)
        parent_output(%allocConv0 : !OutputDistributed0)
        outputs(%allocConv0 : !OutputDistributed0)
        -> !OutputDistributed0 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %alloc1 = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy1 = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                                 outputs(%alloc1 : !InputDistributed)
        -> !InputDistributed

    %allocConv1 = VPURT.AllocDistributed -> !OutputDistributed1
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy1 : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy1 : !InputDistributed)
        parent_output(%allocConv1 : !OutputDistributed1)
        outputs(%allocConv1 : !OutputDistributed1)
        -> !OutputDistributed1 variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputDistributed0, !OutputDistributed1

    // CHECK:       [[NEW_BUFFER:%.+]] =  VPURT.AllocDistributed
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[TILING_COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs([[NEW_BUFFER]] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[TILING_COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:       to !VPUIP.DistributedBuffer<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:        VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:         input([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                     {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED"

    // CHECK:        VPUIP.NCEClusterTask
    // CHECK-SAME:         task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:         input([[SUBVIEW]] : !VPUIP.DistributedBuffer<1x1536x1x1xf16,
    // CHECK-SAME:                                     {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN,
    // CHECK-SAME:                                     {mode = "DUPLICATED", num_clusters = 2 : i64
    // CHECK-SAME:     ->  !VPUIP.DistributedBuffer<1x256x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED"

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputCMXType = memref<1x1536x1x1xf16, #NHWC, @CMX_NN>
!WeightsCMXType = memref<256x1536x1x1xf16, #NHWC, @CMX_NN>
!WTableCMXType = memref<256x1x1x4xsi32, #NCHW, @CMX_NN>
!OutputCMXType = memref<1x256x1x1xf16, #NHWC, @CMX_NN>

// CHECK: @NonDistributedOptimizeSubviewCopyConvPattern
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @NonDistributedOptimizeSubviewCopyConvPattern(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsCMXType,
        %wtable: !WTableCMXType)
         -> (!OutputCMXType, !OutputCMXType){
    %subview0 = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %subview1 = VPUIP.SubView %input [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %alloc0 = memref.alloc(): !InputCMXType
    %copy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%alloc0 : !InputCMXType)
            -> !InputCMXType

    %alloc1 = memref.alloc(): !InputCMXType
    %copy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%alloc1 : !InputCMXType)
            -> !InputCMXType

    %allocConv0 =  memref.alloc(): !OutputCMXType
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%copy0 : !InputCMXType)
        weights(%weights : !WeightsCMXType)
        weight_table(%wtable : !WTableCMXType)
        parent_input(%copy0 : !InputCMXType)
        parent_output(%allocConv0 : !OutputCMXType)
        outputs(%allocConv0 : !OutputCMXType)
        -> !OutputCMXType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 =  memref.alloc(): !OutputCMXType
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%copy1 : !InputCMXType)
        weights(%weights : !WeightsCMXType)
        weight_table(%wtable : !WTableCMXType)
        parent_input(%copy1 : !InputCMXType)
        parent_output(%allocConv1 : !OutputCMXType)
        outputs(%allocConv1 : !OutputCMXType)
        -> !OutputCMXType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputCMXType, !OutputCMXType

    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG0]] : memref<1x3072x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:      outputs(%{{.+}} : memref<1x3072x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x3072x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[COPY]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[COPY]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[SUBVIEW0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[SUBVIEW1]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!RootInputCMXType = memref<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>
!InputCMXType = memref<1x1536x1x1xf16, #NHWC, @CMX_NN>
!WeightsCMXType = memref<256x1536x1x1xf16, #NHWC, @CMX_NN>
!WTableCMXType = memref<256x1x1x4xsi32, #NCHW, @CMX_NN>
!OutputCMXType = memref<1x256x1x1xf16, #NHWC, @CMX_NN>

// CHECK: @NonDistributedOptimizeSubviewCopyConvPatternWithOptimizedCopyIn
// CHECK-SAME: ([[ARG0:%.+]]:  memref<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>
func.func @NonDistributedOptimizeSubviewCopyConvPatternWithOptimizedCopyIn(
        %input: !RootInputCMXType,
        %weights: !WeightsCMXType,
        %wtable: !WTableCMXType)
         -> (!OutputCMXType, !OutputCMXType){

    %alloc_ddr = memref.alloc(): memref<1x3072x1x1xf16, #NHWC, @DDR>
    %spill_copy = VPUIP.Copy inputs(%input : !RootInputCMXType)
                       outputs(%alloc_ddr : memref<1x3072x1x1xf16, #NHWC, @DDR>)
            -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview0 = VPUIP.SubView %spill_copy [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %subview1 = VPUIP.SubView %spill_copy [0, 1536, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %alloc0 = memref.alloc(): !InputCMXType
    %copy0 = VPUIP.Copy inputs(%subview0 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%alloc0 : !InputCMXType)
            -> !InputCMXType

    %alloc1 = memref.alloc(): !InputCMXType
    %copy1 = VPUIP.Copy inputs(%subview1 : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%alloc1 : !InputCMXType)
            -> !InputCMXType

    %allocConv0 =  memref.alloc(): !OutputCMXType
    %conv0 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%copy0 : !InputCMXType)
        weights(%weights : !WeightsCMXType)
        weight_table(%wtable : !WTableCMXType)
        parent_input(%copy0 : !InputCMXType)
        parent_output(%allocConv0 : !OutputCMXType)
        outputs(%allocConv0 : !OutputCMXType)
        -> !OutputCMXType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocConv1 =  memref.alloc(): !OutputCMXType
    %conv1 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%copy1 : !InputCMXType)
        weights(%weights : !WeightsCMXType)
        weight_table(%wtable : !WTableCMXType)
        parent_input(%copy1 : !InputCMXType)
        parent_output(%allocConv1 : !OutputCMXType)
        outputs(%allocConv1 : !OutputCMXType)
        -> !OutputCMXType variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 256], outStart = [0, 0, 128], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv0, %conv1: !OutputCMXType, !OutputCMXType

    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>

    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ARG0]] [0, 1536, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [6144, 1, 6144, 6144]}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[SUBVIEW0]]

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:    task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[SUBVIEW1]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewCopyCDimTooBig
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x8208x1x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewCopyCDimTooBig(
        %input: memref<1x8208x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x8208x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [8208, 1, 8208, 8208]}, @DDR>
    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [8208, 1, 8208, 8208]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %conv: !OutputDistributed

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1] : !VPUIP.DistributedBuffer<1x8208x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x8208x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [8208, 1, 8208, 8208]}, @DDR>

    // CHECK:       VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [8208, 1, 8208, 8208]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithNonCopyConsumer
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewWithNonCopyConsumer(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed, memref<1x1530x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %subview1 = VPUIP.SubView %subview [0, 6, 0, 0] [1, 1530, 1, 1]
    : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    to memref<1x1530x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %conv, %subview1: !OutputDistributed, memref<1x1530x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithCopyConsumerNotToCMX
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewWithCopyConsumerNotToCMX(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed, memref<1x1536x1x1xf16, #NHWC, @DDR>){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %ddrAlloc = memref.alloc() :  memref<1x1536x1x1xf16, #NHWC, @DDR>
    %copy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%ddrAlloc : memref<1x1536x1x1xf16, #NHWC, @DDR>)
    -> memref<1x1536x1x1xf16, #NHWC, @DDR>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %conv, %copy: !OutputDistributed, memref<1x1536x1x1xf16, #NHWC, @DDR>

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> memref<1x1536x1x1xf16, #NHWC, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithDistributedAndNonDistributedCopies
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewWithDistributedAndNonDistributedCopies(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed, memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %cmxAlloc = memref.alloc() :  memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %copy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                       outputs(%cmxAlloc : memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    -> memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %conv, %copy: !OutputDistributed, memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> memref<1x1536x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x2x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x2x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x2x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithNon1x1SpatialSize
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x2x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewWithNon1x1SpatialSize(
        %input: memref<1x3072x2x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 2, 1]
        : memref<1x3072x2x1xf16, #NHWC, @DDR> to memref<1x1536x2x1xf16, {order = #NHWC, strides = [6144, 1, 3072, 3072]}, @DDR>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x2x1xf16, {order = #NHWC, strides = [6144, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 1, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 1, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 1, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 1, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv: !OutputDistributed

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 2, 1] : !VPUIP.DistributedBuffer<1x3072x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 2, 1]
    // CHECK-SAME:       : memref<1x3072x2x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x2x1xf16, {order = #NHWC, strides = [6144, 1, 3072, 3072]}, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x2x1xf16, {order = #NHWC, strides = [6144, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x2x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!PoolWTableDistributed = !VPUIP.DistributedBuffer<
    1536x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

!PoolOutputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithNonConvEndConsumer
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x3072x1x1xf16, #NHWC, @DDR>
func.func @NotOptimizeSubviewWithNonConvEndConsumer(
        %input: memref<1x3072x1x1xf16, #NHWC, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed,
        %wtablePool: !PoolWTableDistributed)
         -> (!OutputDistributed, !PoolOutputDistributed){

    %subview = VPUIP.SubView %input [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %allocPool = VPURT.AllocDistributed -> !PoolOutputDistributed
    %maxpool = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<MAXPOOL>}
        input(%nceTilingCopy : !InputDistributed)
        weight_table(%wtablePool : !PoolWTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocPool : !PoolOutputDistributed)
        outputs(%allocPool : !PoolOutputDistributed)
        -> !PoolOutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    return %conv, %maxpool: !OutputDistributed, !PoolOutputDistributed

    // CHECK-NOT:   VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1] : !VPUIP.DistributedBuffer<1x3072x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED"

    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [1, 1536, 1, 1]
    // CHECK-SAME:       : memref<1x3072x1x1xf16, #NHWC, @DDR>
    // CHECK-SAME:       to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    // CHECK:        VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x1536x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    256x1536x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!WTableDistributed = !VPUIP.DistributedBuffer<
    256x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    alignment = [16, 1, 1, 1]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x256x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    alignment = [1, 16, 1, 1]
}>

// CHECK: @NotOptimizeSubviewWithRMS
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x1x3072xf16, @DDR>
// CHECK-SAME: [[ARG1:%.+]]: memref<3072xf16, @DDR>
func.func @NotOptimizeSubviewWithRMS(
        %input1: memref<1x1x3072xf16, @DDR>,
        %input2: memref<3072xf16, @DDR>,
        %weights: !WeightsDistributed,
        %wtable: !WTableDistributed)
         -> (!OutputDistributed){

    %allocRMSInput1 = memref.alloc() : memref<1x1x3072xf16, [@CMX_NN, 0]>
    %RMSInput1 = VPUIP.Copy inputs(%input1 : memref<1x1x3072xf16, @DDR>) outputs(%allocRMSInput1 : memref<1x1x3072xf16, [@CMX_NN, 0]>) -> memref<1x1x3072xf16, [@CMX_NN, 0]>
    %allocRMSInput2 = memref.alloc() : memref<3072xf16, [@CMX_NN, 0]>
    %RMSInput2 = VPUIP.Copy inputs(%input2 : memref<3072xf16, @DDR>) outputs(%allocRMSInput2 : memref<3072xf16, [@CMX_NN, 0]>) -> memref<3072xf16, [@CMX_NN, 0]>
    %allocRMSOutput = memref.alloc() : memref<1x1x3072xf16, [@CMX_NN, 0]>
    %RMSResult = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS inputs(%RMSInput1 as %arg2: memref<1x1x3072xf16, [@CMX_NN, 0]>, %RMSInput2 as %arg3: memref<3072xf16, [@CMX_NN, 0]>) outputs(%allocRMSOutput as %arg4: memref<1x1x3072xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x3072xf16, [@CMX_NN, 0]>{
      VPUIP.SW.Kernel.run {attrs = [9.9999999747524271E-7]}(%arg2, %arg3, %arg4) : memref<1x1x3072xf16, [@CMX_NN, 0]>, memref<3072xf16, [@CMX_NN, 0]>, memref<1x1x3072xf16, [@CMX_NN, 0]>
    }
    %GenericReshape = VPUIP.GenericReshape inputs(%RMSResult : memref<1x1x3072xf16, [@CMX_NN, 0]>) -> memref<1x3072x1x1xf16, [@CMX_NN, 0]>
    %PermuteCast = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%GenericReshape : memref<1x3072x1x1xf16, [@CMX_NN, 0]>) -> memref<1x3072x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %allocRMSOutputDDR = memref.alloc() : memref<1x3072x1x1xf16, #NHWC, @DDR>
    %RMSResultInDDR = VPUIP.Copy inputs(%PermuteCast : memref<1x3072x1x1xf16, #NHWC, [@CMX_NN, 0]>) outputs(%allocRMSOutputDDR : memref<1x3072x1x1xf16, #NHWC, @DDR>) -> memref<1x3072x1x1xf16, #NHWC, @DDR>

    %subview = VPUIP.SubView %RMSResultInDDR [0, 0, 0, 0] [1, 1536, 1, 1]
        : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>

    %alloc = VPURT.AllocDistributed -> !InputDistributed
    %nceTilingCopy = VPUIP.Copy inputs(%subview : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)
                    outputs(%alloc : !InputDistributed)
        -> !InputDistributed

    %allocConv = VPURT.AllocDistributed -> !OutputDistributed
    %conv = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1], kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>}
        input(%nceTilingCopy : !InputDistributed)
        weights(%weights : !WeightsDistributed)
        weight_table(%wtable : !WTableDistributed)
        parent_input(%nceTilingCopy : !InputDistributed)
        parent_output(%allocConv : !OutputDistributed)
        outputs(%allocConv : !OutputDistributed)
        -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 1535], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    return %conv: !OutputDistributed

    // CHECK:        [[RMS_INPUT1:%.+]] = VPUIP.Copy inputs([[ARG0]] : memref<1x1x3072xf16, @DDR>
    // CHECK:        [[RMS_INPUT2:%.+]] = VPUIP.Copy inputs([[ARG1]] : memref<3072xf16, @DDR>

    // CHECK:        [[RMS_RESULT:%.+]] = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_RMS
    // CHECK-SAME:   inputs([[RMS_INPUT1]] as %{{.+}}: memref<1x1x3072xf16, [@CMX_NN, 0]>, 
    // CHECK-SAME:   [[RMS_INPUT2]] as %{{.+}}: memref<3072xf16, [@CMX_NN, 0]>)

    // CHECK:        [[GENERIC_RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[RMS_RESULT]] : memref<1x1x3072xf16, [@CMX_NN, 0]>)
    // CHECK:        [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERIC_RESHAPE]] : memref<1x3072x1x1xf16, [@CMX_NN, 0]>)
    // CHECK:        [[COPY1:%.+]] = VPUIP.Copy inputs([[PERMUTE_CAST]] : memref<1x3072x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:        [[SUBVIEW:%.+]] = VPUIP.SubView [[COPY1]] [0, 0, 0, 0] [1, 1536, 1, 1] : memref<1x3072x1x1xf16, #NHWC, @DDR> to memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>
    // CHECK:        [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<1x1536x1x1xf16, {order = #NHWC, strides = [3072, 1, 3072, 3072]}, @DDR>)

}
