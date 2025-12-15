//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --mlir-print-elementsattrs-with-hex-if-larger=-1 --unroll-distributed-ops --canonicalize --constant-folding %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x96x8x8xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 8, 8], [1, 32, 8, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 64, 8, 8], [1, 32, 8, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]
}>

!InputSparseMapDistributed = !VPUIP.DistributedBuffer<
    1x96x4x4xi1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 4, 4], [1, 32, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 64, 4, 4], [1, 32, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]
}>

!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x2x4x4xi32, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 4, 4], [1, 1, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 4, 4], [1, 1, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    96x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 16, 1, 1], [32, 16, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]],
    memory_shapes = [[64, 16, 1, 1], [32, 16, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    96x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[64, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]],
    memory_shapes = [[64, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [64, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x96x4x4xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|DUPLICATED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 4, 4], [1, 32, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]],
    memory_shapes = [[1, 96, 4, 4], [1, 96, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!Output_DDR = memref<1x96x4x4xf16, #NHWC, @DDR>

//CHECK-LABEL: @UnrollNceSoKSEPDilatedConv
func.func @UnrollNceSoKSEPDilatedConv() -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %seTable_cst = const.Declare memref<1x2x4x4xi32, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>
            = dense<[[[[36864, 18432, 45056, 22528],
                       [53248, 26624, 61440, 30720],
                       [102400, 51200, 110592, 55296],
                       [118784, 59392, 126976, 63488]],
                      [[167936, 83968, 176128, 88064],
                       [184320, 92160, 192512, 96256],
                       [233472, 116736, 241664, 120832],
                       [249856, 124928, 258048, 129024]]]]> : tensor<1x2x4x4xi32, {order = #NHWC}>
    %seTable_CMX = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 1 : i64} inputs(%seTable_cst : memref<1x2x4x4xi32, #NHWC, @DDR>) outputs(%seTable_CMX : !InputSETableDistributed) -> !InputSETableDistributed
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
                    is_small_kernel_optimized,
                    kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<DWCONV>
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
                        inEnd = [3, 3, 63], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [3, 3, 63], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [3, 3, 31], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [3, 3, 31], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %parent_out: !Output_DDR

    // CHECK:      [[CST_CL_0:%.+]] = const.Declare memref<1x1x4x4xi32, #NHWC, @DDR> = dense<
    // CHECK-SAME{LITERAL}:       [36864, 45056, 53248, 61440]
    // CHECK-SAME{LITERAL}:       [102400, 110592, 118784, 126976]
    // CHECK-SAME{LITERAL}:       [167936, 176128, 184320, 192512]
    // CHECK-SAME{LITERAL}:       [233472, 241664, 249856, 258048]
    // CHECK-SAME{LITERAL}:  ]> : tensor<1x1x4x4xi32, {order = #NHWC}>

    // CHECK:      [[CST_CL_1:%.+]] = const.Declare memref<1x1x4x4xi32, #NHWC, @DDR> = dense<
    // CHECK-SAME{LITERAL}:     [18432, 22528, 26624, 30720]
    // CHECK-SAME{LITERAL}:     [51200, 55296, 59392, 63488]
    // CHECK-SAME{LITERAL}:     [83968, 88064, 92160, 96256]
    // CHECK-SAME{LITERAL}:     [116736, 120832, 124928, 129024]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x4x4xi32, {order = #NHWC}>

    // CHECK: VPUIP.NNDMA
    // CHECK-SAME: inputs([[CST_CL_0]]
    // CHECK-SAME: [@CMX_NN, 0]

    // CHECK: VPUIP.NNDMA
    // CHECK-SAME: inputs([[CST_CL_1]]
    // CHECK-SAME: [@CMX_NN, 1]
}
