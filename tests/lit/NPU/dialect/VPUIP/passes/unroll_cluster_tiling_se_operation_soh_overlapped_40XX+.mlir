//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --unroll-cluster-tiling --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x5x5xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 5], [1, 16, 2, 5]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>

!InputSparseMapDistributed = !VPUIP.DistributedBuffer<
    1x16x22x22xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 11, 22], [1, 16, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 16, 12, 22], [1, 16, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x1x22x22xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 11, 22], [1, 1, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 1, 12, 22], [1, 1, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x10x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
}>

!InputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!InputSparseMapStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!OutputSparsityStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

!Output_DDR = memref<1x16x10x10xf16, #NHWC, @DDR>

//CHECK-LABEL: @UnrollNceSoHSEPInterpolateOverlappedTwoClusters
func.func @UnrollNceSoHSEPInterpolateOverlappedTwoClusters() -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %seTable_cst = const.Declare memref<1x1x22x22xi32, #NHWC, @DDR> = dense<[[[
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
        ]]]> : tensor<1x1x22x22xi32, {order = #NHWC}>
    %seTable_CMX = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, #NHWC, @DDR>) outputs(%seTable_CMX : !InputSETableDistributed) -> !InputSETableDistributed
    }

    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_input_sparsity_map = VPURT.DeclareBuffer <CMX_NN> <31264> -> !InputSparseMapDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31008> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed
    %parent_out_sparsity_map = VPURT.DeclareBuffer <CMX_NN> <103138> -> !OutputDistributed

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }   input(%parent_input_cmx : !InputDistributed)
                input_sparsity_map(%parent_input_sparsity_map : !InputSparseMapDistributed)
                input_storage_element_table(%seTable_CMX: !InputSETableDistributed)
                weights(%weights : !WeightsDistributed)
                weight_table(%weights_table : !WeightsTableDistributed)
                parent_input(%parent_input_cmx : !InputDistributed)
                parent_input_sparsity_map(%parent_input_sparsity_map : !InputSparseMapDistributed)
                parent_input_storage_element_table(%seTable_CMX: !InputSETableDistributed)
                parent_output(%parent_out_cmx : !OutputDistributed)
                parent_output_sparsity_map(%parent_out_sparsity_map : !OutputDistributed)
                outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %parent_out: !Output_DDR

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, #NHWC, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, #NHWC, @DDR> = dense<[[[
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x5x5xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 5], [1, 16, 2, 5]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>

!InputSparseMapDistributed = !VPUIP.DistributedBuffer<
    1x16x22x22xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 11, 22], [1, 16, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 16, 12, 22], [1, 16, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x1x22x22xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 11, 22], [1, 1, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 1, 12, 22], [1, 1, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 1, 1, 4], [16, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x10x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
}>

!InputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!InputSparseMapStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!OutputSparsityStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

!Output_DDR = memref<1x16x10x10xf16, #NHWC, @DDR>

//CHECK-LABEL: @UnrollNceSoHSEPInterpolateOverlappedTwoClustersAndTwoUsers
func.func @UnrollNceSoHSEPInterpolateOverlappedTwoClustersAndTwoUsers() -> (!Output_DDR, !Output_DDR) {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %seTable_cst = const.Declare memref<1x1x22x22xi32, #NHWC, @DDR> = dense<[[[
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
        ]]]> : tensor<1x1x22x22xi32, {order = #NHWC}>

    // First NCE user
    %seTable_CMX_0 = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, #NHWC, @DDR>) outputs(%seTable_CMX_0 : !InputSETableDistributed) -> !InputSETableDistributed
    }

    %parent_out_0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %parent_input_cmx_0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_input_sparsity_map_0 = VPURT.DeclareBuffer <CMX_NN> <31264> -> !InputSparseMapDistributed
    %weights_0 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %weights_table_0 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31008> -> !WeightsTableDistributed
    %parent_out_cmx_0 = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed
    %parent_out_sparsity_map_0 = VPURT.DeclareBuffer <CMX_NN> <103138> -> !OutputDistributed

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }   input(%parent_input_cmx_0 : !InputDistributed)
                input_sparsity_map(%parent_input_sparsity_map_0 : !InputSparseMapDistributed)
                input_storage_element_table(%seTable_CMX_0: !InputSETableDistributed)
                weights(%weights_0 : !WeightsDistributed)
                weight_table(%weights_table_0 : !WeightsTableDistributed)
                parent_input(%parent_input_cmx_0 : !InputDistributed)
                parent_input_sparsity_map(%parent_input_sparsity_map_0 : !InputSparseMapDistributed)
                parent_input_storage_element_table(%seTable_CMX_0: !InputSETableDistributed)
                parent_output(%parent_out_cmx_0 : !OutputDistributed)
                parent_output_sparsity_map(%parent_out_sparsity_map_0 : !OutputDistributed)
                outputs(%parent_out_cmx_0 : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx_0: !OutputDistributed) outputs(%parent_out_0: !Output_DDR) -> !Output_DDR
    }

    // Second NCE user
    %seTable_CMX_1 = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, #NHWC, @DDR>) outputs(%seTable_CMX_1 : !InputSETableDistributed) -> !InputSETableDistributed
    }

    %parent_out_1 = VPURT.DeclareBuffer <NetworkOutput> [10240] <0> -> !Output_DDR

    %parent_input_cmx_1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_input_sparsity_map_1 = VPURT.DeclareBuffer <CMX_NN> <31264> -> !InputSparseMapDistributed
    %parent_input_se_table_1 = VPURT.DeclareBuffer <CMX_NN> <33442> -> !InputSETableDistributed
    %weights_1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %weights_table_1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31008> -> !WeightsTableDistributed
    %parent_out_cmx_1 = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed
    %parent_out_sparsity_map_1 = VPURT.DeclareBuffer <CMX_NN> <103138> -> !OutputDistributed

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }   input(%parent_input_cmx_1 : !InputDistributed)
                input_sparsity_map(%parent_input_sparsity_map_1 : !InputSparseMapDistributed)
                input_storage_element_table(%seTable_CMX_1 : !InputSETableDistributed)
                weights(%weights_1 : !WeightsDistributed)
                weight_table(%weights_table_1 : !WeightsTableDistributed)
                parent_input(%parent_input_cmx_1 : !InputDistributed)
                parent_input_sparsity_map(%parent_input_sparsity_map_1 : !InputSparseMapDistributed)
                parent_input_storage_element_table(%seTable_CMX_1: !InputSETableDistributed)
                parent_output(%parent_out_cmx_1 : !OutputDistributed)
                parent_output_sparsity_map(%parent_out_sparsity_map_1 : !OutputDistributed)
                outputs(%parent_out_cmx_1 : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx_1: !OutputDistributed) outputs(%parent_out_1: !Output_DDR) -> !Output_DDR
    }

    return %parent_out_0, %parent_out_1: !Output_DDR, !Output_DDR

    // This test is to check the same SETable constant is used by two different NCE ops
    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, #NHWC, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, #NHWC, @DDR> = dense<[[[
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [1, 1, 1, 1, 1, 1025, 1025, 1025, 1025, 2049, 2049, 2049, 2049, 3073, 3073, 3073, 3073, 4097, 4097, 4097, 4097, 4097]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK-SAME:      [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>
}
