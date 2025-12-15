//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 2],
    num_clusters = 3,
    uniform_distributed_segments
}>

!InputSparseMapDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 3
}>

!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 3
}>

!OutputSparseMapDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xi1, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 2],
    num_clusters = 3
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x33x33xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x33x33xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!InputSparseMapStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!OutputSparsityStub_CMX = memref<1x16x33x33xi1, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// Duplicate of UnrollNceSoHOutputOverlapped from unroll_cluster_tiling_soh_overlapped_40XX+.mlir,
// but verifies selection of candidates for split. 50XX has only one DMA ports, so no candidates
//CHECK-LABEL: @UnrollNceSoHOutputOverlappedSplitCandidates
func.func @UnrollNceSoHOutputOverlappedSplitCandidates(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare !WeightsTable_DDR = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <26400> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <31008> -> !WeightsTableDistributed

    %parent_input_sparsity_map = VPURT.DeclareBuffer <CMX_NN> <31264> -> !InputSparseMapDistributed
    %parent_input_se_table = VPURT.DeclareBuffer <CMX_NN> <33442> -> !InputSETableDistributed
    %parent_out_sparsity_map = VPURT.DeclareBuffer <CMX_NN> <103138> -> !OutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }
    // Select cluster 2 as candidate for split
    // CHECK:   VPUIP.NNDMA {port = 0 : i64, split_candidate}
    // CHECK-SAME:      memref<1x16x12x33xf16, #NHWC, @DDR>
    // CHECK-SAME:      memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1:2 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
            }   input(%parent_input_cmx : !InputDistributed)
                input_sparsity_map(%parent_input_sparsity_map : !InputSparseMapDistributed)
                input_storage_element_table(%parent_input_se_table : !InputSETableDistributed)
                weights(%weights : !WeightsDistributed)
                weight_table(%weights_table : !WeightsTableDistributed)
                parent_input(%parent_input_cmx : !InputDistributed)
                parent_input_sparsity_map(%parent_input_sparsity_map : !InputSparseMapDistributed)
                parent_input_storage_element_table(%parent_input_se_table : !InputSETableDistributed)
                parent_output(%parent_out_cmx : !OutputDistributed)
                parent_output_sparsity_map(%parent_out_sparsity_map : !OutputDistributed)
                outputs(%parent_out_cmx : !OutputDistributed)
                output_sparsity_map(%parent_out_sparsity_map : !OutputDistributed)
                        -> !OutputDistributed, !OutputDistributed
                variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [32, 10, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 11, 0], outEnd = [32, 21, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    DPUTask {
                        outStart = [0, 22, 0], outEnd = [32, 32, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
                        cluster_id = 2 : i64
                    }
                } PPE :  {
                }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }
    // Select cluster 2 as candidate for split
    // CHECK:   VPUIP.NNDMA {port = 0 : i64, split_candidate}
    // CHECK-SAME:      memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>
    // CHECK-SAME:      memref<1x16x11x33xf16, #NHWC, @DDR>

    return %output: !Output_DDR
}
