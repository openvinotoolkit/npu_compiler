//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// In this example DPU task in SOHoverlapped mode with input 1x16x33x33xf16
// will be split into 3 clusters where input 0 & 2 will be 1x16x12x33
// and input 1 will be 1x16x13x33
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

//CHECK-LABEL: @UnrollNceSoHOutputOverlapped
func.func @UnrollNceSoHOutputOverlapped(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
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

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC, @DDR>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x12x33xf16, #NHWC, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <10560> -> memref<1x16x13x33xf16, #NHWC, @DDR>
    //CHECK:    [[IN3_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <22176> -> memref<1x16x12x33xf16, #NHWC, @DDR>

    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x13x33xf16, #NHWC, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <12672> -> memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK:    [[OUT3_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <23232> -> memref<1x16x11x33xf16, #NHWC, @DDR>

    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <13728> -> memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <13728> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <13728> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>

    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <13728> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK-NEXT: inwardHaloRegions = [
    //CHECK-NEXT:   #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:  ]
    //CHECK-NOT:  outwardHaloRegions

    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <13728> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK-NEXT: inwardHaloRegions = [
    //CHECK-NEXT:   #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 33], offset = [0, 0, 10, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT: outwardHaloRegions
    //CHECK-NEXT:   shape = [1, 16, 2, 33]
    //CHECK-SAME:   offset = [0, 0, 0, 0]
    //CHECK-SAME:   cluster_id = 1 : i64
    //CHECK-SAME:   inwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]

    //CHECK:    [[OUT3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <13728> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK-NOT:  inwardHaloRegions
    //CHECK-NEXT: outwardHaloRegions
    //CHECK-NEXT:   shape = [1, 16, 1, 33]
    //CHECK-SAME:   offset = [0, 0, 0, 0]
    //CHECK-SAME:   cluster_id = 2 : i64
    //CHECK-SAME:   inwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 33], offset = [0, 0, 10, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]

    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <26400> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <26400> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <26400> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <26400> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <31008> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <31008> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <31008> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <31008> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:    [[IN_SPARSE_MAP_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <31264> -> memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN_SPARSE_MAP_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <31264> -> memref<1x16x13x33xi1, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN_SPARSE_MAP_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <31264> -> memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 2]>
    //CHECK:    [[IN_SE_TABLE_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33442> -> memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN_SE_TABLE_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33442> -> memref<1x16x13x33xi32, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN_SE_TABLE_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <33442> -> memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 2]>

    //CHECK:    [[OUT_SPARSE_MAP_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <103138> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK-NEXT: inwardHaloRegions = [
    //CHECK-NEXT:   #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>
    //CHECK-NEXT: ]>
    //CHECK-NOT:  outwardHaloRegions

    //CHECK:    [[OUT_SPARSE_MAP_2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <103138> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK-NEXT: inwardHaloRegions = [
    //CHECK-NEXT:   #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 33], offset = [0, 0, 10, 0], cluster_id = 1 : i64>
    //CHECK-NEXT: ]
    //CHECK-NEXT: outwardHaloRegions = [
    //CHECK-NEXT:    #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:    shape = [1, 16, 2, 33],
    //CHECK-SAME:    offset = [0, 0, 0, 0],
    //CHECK-SAME:    cluster_id = 1 : i64,
    //CHECK-SAME:    inwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]>

    //CHECK:    [[OUT_SPARSE_MAP_3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <103138> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT: 1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2],
    //CHECK-NEXT: outwardHaloRegions = [
    //CHECK-NEXT:    #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:    shape = [1, 16, 1, 33],
    //CHECK-SAME:    offset = [0, 0, 0, 0],
    //CHECK-SAME:    cluster_id = 2 : i64,
    //CHECK-SAME:    inwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 33], offset = [0, 0, 10, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]>


    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x12x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x13x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN3_DDR]] : memref<1x16x12x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN3_CMX_COPY]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           input_sparsity_map([[IN_SPARSE_MAP_1]] : memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_1]] : memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input_sparsity_map([[IN_SPARSE_MAP_1]] : memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_1]] : memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                parent_output_sparsity_map([[OUT_SPARSE_MAP_1]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                output_sparsity_map([[OUT_SPARSE_MAP_1]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 10, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           input_sparsity_map([[IN_SPARSE_MAP_2]] : memref<1x16x13x33xi1, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_2]] : memref<1x16x13x33xi32, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input_sparsity_map([[IN_SPARSE_MAP_2]] : memref<1x16x13x33xi1, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_2]] : memref<1x16x13x33xi32, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            parent_output_sparsity_map([[OUT_SPARSE_MAP_2]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            output_ITI_buff([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:            outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            output_sparsity_map([[OUT_SPARSE_MAP_2]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 21, 15], outStart = [0, 11, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN3_CMX]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           input_sparsity_map([[IN_SPARSE_MAP_3]] : memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_3]] : memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE3_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x12x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input_sparsity_map([[IN_SPARSE_MAP_3]] : memref<1x16x12x33xi1, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_3]] : memref<1x16x12x33xi32, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:                parent_output_sparsity_map([[OUT_SPARSE_MAP_3]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:                output_ITI_buff([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:                outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:                output_sparsity_map([[OUT_SPARSE_MAP_3]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 32, 15], outStart = [0, 22, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x13x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x13x33xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x11x33xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT3_CMX_COPY]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:       outputs([[OUT3_DDR]] : memref<1x16x11x33xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x33xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!InputDistributed = !VPUIP.DistributedBuffer<
                1x3x224x224xf16,
                #NCHW,
                @CMX_NN, {
                    mode = "OVERLAPPED",
                    num_tiles = [1, 1, 2, 1],
                    kernel = [7, 7],
                    pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
                    strides = [2, 2],
                    num_clusters = 2 : i64,
                    uniform_distributed_segments
                }>

!OutputDistributed = !VPUIP.DistributedBuffer<
        1x224x4x224x!qElemType,
        #NWCH,
        @CMX_NN, {
            mode = "OVERLAPPED",
            num_tiles = [1, 1, 1, 2],
            kernel = [7, 7],
            pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            strides = [2, 2],
            num_clusters = 2 : i64,
            uniform_distributed_segments,
            equal_memory_and_compute_view
        }
    >

!QuantizedDistributed = !VPUIP.DistributedBuffer<
        1x224x4x224x!qElemType,
        #NWCH,
        @CMX_NN, {
            mode = "OVERLAPPED",
            num_tiles = [1, 1, 1, 2],
            kernel = [7, 7],
            pads = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            strides = [2, 2],
            num_clusters = 2 : i64,
            uniform_distributed_segments,
            equal_memory_and_compute_view
        }
    >

!OutputType_DDR = memref<1x224x4x224x!qElemType, {order = #NWCH}, @DDR>

// CHECK-LABEL: @OmitHaloRegionsForEqualMemView
func.func @OmitHaloRegionsForEqualMemView(%arg0: memref<1x3x224x224xf16, @DDR>,
                                          %arg1: memref<1x224x4x224x!qElemType, #NWCH, @DDR>)
        -> memref<1x224x4x224x!qElemType, #NWCH, @DDR> {
    %NET_INPUT = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x224x224xf16, @DDR>
    %CMX_INPUT = VPURT.DeclareBuffer <CMX_NN> <512> -> !InputDistributed
    %CMX_OUTPUT = VPURT.DeclareBuffer <CMX_NN> <155072> -> !OutputDistributed
    %QUANTIZE_INPUT = VPURT.DeclareBuffer <CMX_NN> <512> -> !QuantizedDistributed

    %WAIT_INPUT_COPY = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %WAIT_QUANTIZE = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %WAIT_OUTPUT_COPY = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // NNDMA for input
    VPURT.Task updates(%WAIT_INPUT_COPY : !VPURT.Barrier) attributes {
        isTrailingSWLayer = false
    } {
        %31 = VPUIP.NNDMA {
            port = 0 : i64
        }
        inputs(%NET_INPUT : memref<1x3x224x224xf16, @DDR>)
        outputs(%CMX_INPUT : !InputDistributed)
            -> !InputDistributed
    }

    // DPU task
    VPURT.Task
        waits(%WAIT_INPUT_COPY : !VPURT.Barrier)
        updates(%WAIT_QUANTIZE : !VPURT.Barrier)
        attributes {isTrailingSWLayer = false} {

        %31 = VPUIP.NCEClusterTask {
            is_permute_quantize,
            is_superdense,
            minimumHardwareExecutionCost = 4294967300 : i64,
            task_type = #VPUIP.nce_task_type<ELTWISE>
        }
        input(%QUANTIZE_INPUT : !QuantizedDistributed)
        weights(%QUANTIZE_INPUT : !QuantizedDistributed)
        parent_input(%QUANTIZE_INPUT : !QuantizedDistributed)
        parent_output(%CMX_OUTPUT : !OutputDistributed)
        outputs(%CMX_OUTPUT : !OutputDistributed)
            -> !OutputDistributed variants : {
          DPUTask {
              cluster_id = 0 : i64,
              inEnd = [113, 2, 223],
              inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
              outEnd = [113, 2, 223],
              outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
          }
          DPUTask {
              cluster_id = 1 : i64,
              inEnd = [114, 2, 223],
              inStart = [0, 0, 0],
              mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
              outEnd = [114, 2, 223],
              outStart = [0, 0, 0],
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
          }
        } PPE : {
          PPETask {
              ppe = #VPU.PPEStub<>
          }
        }
    }

    %NET_OUT = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !OutputType_DDR

    // NNDMA for output
    VPURT.Task
        waits(%WAIT_QUANTIZE : !VPURT.Barrier)
        updates(%WAIT_OUTPUT_COPY : !VPURT.Barrier)
        attributes {
            isTrailingSWLayer = false
        } {
        %31 = VPUIP.NNDMA
            inputs(%CMX_OUTPUT : !OutputDistributed)
            outputs(%NET_OUT : !OutputType_DDR)
                -> !OutputType_DDR
    }

    return %arg1 : memref<1x224x4x224x!qElemType, #NWCH, @DDR>

    // CHECK-NOT: inwardHaloRegions
    // CHECK-NOT: outwardHaloRegions
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<1x16x112x112xi1,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [5, 5], pads = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<1x16x112x112xi1, {allocSize = 13536 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x112x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
// CHECK-LABEL: @UnrollSpillCompressCandidates
func.func @UnrollSpillCompressCandidates(%output: !DistBuf) -> !DistBuf {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !DistBuf
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !DistBuf

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %47 = VPUIP.NNDMA {port = 0 : i64, compress_candidate, spillId = 0 : i64} inputs(%13 : !DistBuf) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) {
        %47 = VPUIP.NNDMA {port = 0 : i64, compress_candidate, spillId = 0 : i64} inputs(%17 : !Compact_DDR) outputs(%18 : !DistBuf) -> !DistBuf
    }
    return %18 : !DistBuf

    // Check per cluster offset are properly set to take into account swizzling and worst case size with compress DMA
    // CHECK: [[DDR0_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {allocSize = 13536 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK: [[DDR0_1:%.*]] = VPURT.DeclareBuffer <DDR> <13536> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {allocSize = 13536 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK: [[DDR1_0:%.*]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {allocSize = 13536 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    // CHECK: [[DDR1_1:%.*]] = VPURT.DeclareBuffer <DDR> <13536> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {allocSize = 13536 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<
    1x48x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>},
    @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 48, 4, 8], [1, 48, 4, 8]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0]],
    memory_shapes = [[1, 48, 5, 8], [1, 48, 5, 8]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]]
}>

!Compact_DDR = memref<1x48x8x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 4096 : i64>}, @DDR>

func.func @AdjustSizeAlignmentkOverlapedCMXToDDR(%output: !Compact_DDR) -> !Compact_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %18 = VPURT.DeclareBuffer <CMX_NN> <1280> {swizzlingKey = 5 : i64} -> !DistBuf
    %19 = VPURT.DeclareBuffer <CMX_NN> <5376> {swizzlingKey = 5 : i64} -> !DistBuf
    %20 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA {port = 0 : i64, spillId = 2 : i64} inputs(%18 : !DistBuf) outputs(%20 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA {port = 1 : i64, spillId = 2 : i64} inputs(%19 : !DistBuf) outputs(%20 : !Compact_DDR) -> !Compact_DDR
    }

    return %20 : !Compact_DDR

    // Check alignment is set back to 512 for per cluster buffers
    // CHECK: [[DDR0_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <1280> {swizzlingKey = 5 : i64} -> memref<1x48x5x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
    // CHECK: [[DDR0_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <1280> {swizzlingKey = 5 : i64} -> memref<1x48x5x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 1]>
    // CHECK: [[DDR0_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <5376> {swizzlingKey = 5 : i64} -> memref<1x48x5x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 0]>
    // CHECK: [[DDR0_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <5376> {swizzlingKey = 5 : i64} -> memref<1x48x5x8xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, [@CMX_NN, 1]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<
    1x48x16x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>},
    @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    compute_shapes = [[1, 48, 4, 16], [1, 48, 4, 16], [1, 48, 4, 16], [1, 48, 4, 16]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 4, 0], [0, 0, 8, 0], [0, 0, 12, 0]],
    memory_shapes = [[1, 48, 5, 16], [1, 48, 6, 16], [1, 48, 6, 16], [1, 48, 5, 16]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 3, 0], [0, 0, 7, 0], [0, 0, 11, 0]]
}>

!Compact_DDR = memref<1x48x16x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 17408 : i64>}, @DDR>

// When spill segmented or overlapped distributed buffer to DDR, the DDR offset need be calculated according to all clusters' individual shapes,
// because clusters may not allocate the same size
// CHECK-LABEL: @DifferentSizeOverlapedCMXToDDR
func.func @DifferentSizeOverlapedCMXToDDR(%output: !Compact_DDR) -> !Compact_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %18 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !DistBuf
    %20 = VPURT.DeclareBuffer <DDR> <107840> {swizzlingKey = 5 : i64} -> !Compact_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA {port = 0 : i64, spillId = 2 : i64} inputs(%18 : !DistBuf) outputs(%20 : !Compact_DDR) -> !Compact_DDR
    }

    return %20 : !Compact_DDR

    // CHECK: [[DDR_0:%.*]] = VPURT.DeclareBuffer <DDR> <107840> {swizzlingKey = 5 : i64} -> memref<1x48x5x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
    // CHECK: [[DDR_1:%.*]] = VPURT.DeclareBuffer <DDR> <116032> {swizzlingKey = 5 : i64} -> memref<1x48x6x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
    // CHECK: [[DDR_2:%.*]] = VPURT.DeclareBuffer <DDR> <125248> {swizzlingKey = 5 : i64} -> memref<1x48x6x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
    // CHECK: [[DDR_3:%.*]] = VPURT.DeclareBuffer <DDR> <134464> {swizzlingKey = 5 : i64} -> memref<1x48x5x16xf16, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!OutputDistributed = !VPUIP.DistributedBuffer<
        1x784x32x6xf16,
        #NWCH,
        @CMX_NN, {
            mode = "OVERLAPPED",
            num_tiles = [1, 1, 1, 4],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[1, 784, 32, 2], [1, 784, 32, 2], [1, 784, 32, 1], [1, 784, 32, 1]],
            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 5]],
            memory_shapes = [[1, 784, 32, 3], [1, 784, 32, 3], [1, 784, 32, 3], [1, 784, 32, 3]],
            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]]
            }
    >

!InputDistributed = !VPUIP.DistributedBuffer<
        1x784x32x6xf16,
        #NHWC,
        @CMX_NN, {
            mode = "OVERLAPPED",
            num_tiles = [1, 1, 1, 4],
            num_clusters = 4 : i64,
            uniform_distributed_segments,
            compute_shapes = [[1, 784, 32, 2], [1, 784, 32, 2], [1, 784, 32, 1], [1, 784, 32, 1]],
            compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 5]],
            memory_shapes = [[1, 784, 32, 2], [1, 784, 32, 2], [1, 784, 32, 1], [1, 784, 32, 1]],
            memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 5]]
            }
    >

//CHECK-LABEL: @UnrollNceOutputOverlappedHaloOverNonadjacentCluster
func.func @UnrollNceOutputOverlappedHaloOverNonadjacentCluster() -> memref<1x784x32x6xf16, #NWCH, @DDR> {
  %BAR0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %CONV_INPUT = VPURT.DeclareBuffer <CMX_NN> <151104> -> !InputDistributed
  %CONV_OUTPUT = VPURT.DeclareBuffer <CMX_NN> <576> -> !OutputDistributed
  %OUTPUT = VPURT.DeclareBuffer <DDR> <0> -> memref<1x784x32x6xf16, #NWCH, @DDR>
  VPURT.Task updates(%BAR0 : !VPURT.Barrier) {
    %1 = VPUIP.NCEClusterTask {is_permute_quantize, minimumHardwareExecutionCost = 4751 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
      input(%CONV_INPUT : !InputDistributed)
      weights(%CONV_INPUT : !InputDistributed)
      parent_input(%CONV_INPUT : !InputDistributed)
      parent_output(%CONV_OUTPUT : !OutputDistributed)
      outputs(%CONV_OUTPUT  : !OutputDistributed)
        -> !OutputDistributed
      variants : {
         DPUTask {cluster_id = 0 : i64, inEnd = [1, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 31, 783], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
         DPUTask {cluster_id = 1 : i64, inEnd = [1, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [1, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
         DPUTask {cluster_id = 2 : i64, inEnd = [0, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
         DPUTask {cluster_id = 3 : i64, inEnd = [0, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
  }
  VPURT.Task waits(%BAR0 : !VPURT.Barrier) {
    %1 = VPUIP.NNDMA inputs(%CONV_OUTPUT : !OutputDistributed) outputs(%OUTPUT : memref<1x784x32x6xf16, #NWCH, @DDR>) -> memref<1x784x32x6xf16, #NWCH, @DDR>
  }
  return %OUTPUT : memref<1x784x32x6xf16, #NWCH, @DDR>

  //CHECK: [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  //CHECK: [[WEIGHT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <151104> -> memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 0]>
  //CHECK: [[WEIGHT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <151104> -> memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 1]>
  //CHECK: [[WEIGHT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <151104> -> memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 2]>
  //CHECK: [[WEIGHT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <151104> -> memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 3]>
  //CHECK: [[IN0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <151104> -> memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 0]>
  //CHECK: [[IN1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <151104> -> memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 1]>
  //CHECK: [[IN2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <151104> -> memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 2]>
  //CHECK: [[IN3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <151104> -> memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 3]>
  //CHECK: [[COPY_IN0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 0]>
  //CHECK: [[COPY_IN1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 1]>
  //CHECK: [[COPY_IN2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 2]>
  //CHECK: [[COPY_IN3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 3]>
  //CHECK: [[OUT0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> !VPUIP.ITIBuffer<
  //CHECK-NEXT:             1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 0],
  //CHECK-NEXT:              inwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 2], cluster_id = 0 : i64>
  //CHECK-NEXT:             ],
  //CHECK-NEXT:              outwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.OutwardHaloRegionAttr<
  //CHECK-SAME:                  shape = [1, 784, 32, 1],
  //CHECK-SAME:                  offset = [0, 0, 0, 1],
  //CHECK-SAME:                  cluster_id = 0 : i64,
  //CHECK-SAME:                  inwardHaloRegions = [
  //CHECK-NEXT:                     #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
  //CHECK-NEXT:                 ]>
  //CHECK: [[OUT1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> !VPUIP.ITIBuffer<
  //CHECK-NEXT:              1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 1],
  //CHECK-NEXT:              inwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
  //CHECK-NEXT:               ],
  //CHECK-NEXT:              outwardHaloRegions = [
  //CHECK-NEXT:                               #VPUIP.OutwardHaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 1], cluster_id = 1 : i64, inwardHaloRegions = [
  //CHECK-NEXT:                                #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 2], cluster_id = 0 : i64>
  //CHECK-NEXT:                          ]>,
  //CHECK-NEXT:                       #VPUIP.OutwardHaloRegionAttr<shape = [1, 784, 32, 2], offset = [0, 0, 0, 1], cluster_id = 1 : i64, inwardHaloRegions = [
  //CHECK-NEXT:                          #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 2], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
  //CHECK-NEXT:                   ]>
  //CHECK-NEXT:                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 2], cluster_id = 1 : i64, inwardHaloRegions = [
  //CHECK-NEXT:                   #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
  //CHECK-NEXT:             ]>
  //CHECK: [[OUT2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> !VPUIP.ITIBuffer<
  //CHECK-NEXT:              1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 2],
  //CHECK-NEXT:              inwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 2], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
  //CHECK-NEXT:              ],
  //CHECK-NEXT:              outwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.OutwardHaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 2], cluster_id = 2 : i64, inwardHaloRegions = [
  //CHECK-NEXT:                     #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 1], cluster_id = 3 : i64>
  //CHECK-NEXT:             ]>
  //CHECK: [[OUT3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> !VPUIP.ITIBuffer<
  //CHECK-NEXT:             1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 3],
  //CHECK-NEXT:              inwardHaloRegions = [
  //CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 0], cluster_id = 3 : i64>,
  //CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 784, 32, 1], offset = [0, 0, 0, 1], cluster_id = 3 : i64>
  //CHECK-NEXT:              ]>
  //CHECK: [[DDR_OUT:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x784x32x6xf16, #NWCH, @DDR>
  //CHECK: [[COPY_OUT0:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK: [[COPY_OUT1:%.*]] = VPURT.DeclareBuffer <DDR> <50176> -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK: [[COPY_OUT2:%.*]] = VPURT.DeclareBuffer <DDR> <100352> -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK: [[COPY_OUT3:%.*]] = VPURT.DeclareBuffer <DDR> <150528> -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK:              VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                  VPUIP.NCEClusterTask {is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}
  //CHECK-SAME:                 input([[IN0]] : memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 0]>)
  //CHECK-SAME:                 weights([[WEIGHT0]] : memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 0]>)
  //CHECK:                      output_ITI_buff([[OUT1]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 1]
  //CHECK:                      outputs([[OUT0]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 0]
  //CHECK:                 variants : {
  //CHECK:                           DPUTask {cluster_id = 0 : i64, inEnd = [1, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [1, 31, 783], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  //CHECK:                      PPE : {
  //CHECK:                           PPETask {ppe = #VPU.PPEStub<>}
  //CHECK:                      }
  //CHECK:               }
  //CHECK:              VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                  VPUIP.NCEClusterTask {is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}
  //CHECK-SAME:                 input([[IN1]] : memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 1]>)
  //CHECK-SAME:                 weights([[WEIGHT1]] : memref<1x784x32x2xf16, #NHWC, [@CMX_NN, 1]>)
  //CHECK:                      output_ITI_buff([[OUT0]], [[OUT2]], [[OUT3]]
  //CHECK:                 outputs([[OUT1]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 1]
  //CHECK:                  variants : {
  //CHECK:                           DPUTask {cluster_id = 1 : i64, inEnd = [1, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [1, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  //CHECK:                      PPE : {
  //CHECK:                           PPETask {ppe = #VPU.PPEStub<>}
  //CHECK:                      }
  //CHECK:               }
  //CHECK:              VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                  VPUIP.NCEClusterTask {is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}
  //CHECK-SAME:                 input([[IN2]] : memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 2]>)
  //CHECK-SAME:                 weights([[WEIGHT2]] : memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 2]>)
  //CHECK:                      output_ITI_buff([[OUT3]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 3],
  //CHECK:                      outputs([[OUT2]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 2]
  //CHECK:                     variants : {
  //CHECK:                           DPUTask {cluster_id = 2 : i64, inEnd = [0, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  //CHECK:                      PPE : {
  //CHECK:                           PPETask {ppe = #VPU.PPEStub<>}
  //CHECK:                      }
  //CHECK:               }
  //CHECK:              VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                  VPUIP.NCEClusterTask {is_permute_quantize, task_type = #VPUIP.nce_task_type<ELTWISE>}
  //CHECK-SAME:                 input([[IN3]] : memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 3]>)
  //CHECK-SAME:                 weights([[WEIGHT3]] : memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 3]>)
  //CHECK-SAME:                 parent_input([[IN3]] : memref<1x784x32x1xf16, #NHWC, [@CMX_NN, 3]>)
  //CHECK-SAME:                 parent_output([[OUT3]] : !VPUIP.ITIBuffer<
  //CHECK-NEXT:                     1x784x32x3xf16, {order = #NWCH}, [@CMX_NN, 3]
  //CHECK:                      variants : {
  //CHECK:                          DPUTask {cluster_id = 3 : i64, inEnd = [0, 31, 783], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [2, 31, 783], outStart = [2, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
  //CHECK:                      PPE : {
  //CHECK:                           PPETask {ppe = #VPU.PPEStub<>}
  //CHECK:                      }
  //CHECK:               }

  //CHECK:              VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                 VPUIP.NNDMA
  //CHECK-SAME:                 inputs([[COPY_IN0]] : memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 0]>) outputs([[COPY_OUT0]] : memref<1x784x32x3xf16, #NWCH, @DDR>) -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK:              }
  //CHECK:              VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                 VPUIP.NNDMA
  //CHECK-SAME:                 inputs([[COPY_IN1]] : memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 1]>) outputs([[COPY_OUT1]] : memref<1x784x32x3xf16, #NWCH, @DDR>) -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK:              }
  //CHECK:              VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                 VPUIP.NNDMA
  //CHECK-SAME:                 inputs([[COPY_IN2]] : memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 2]>) outputs([[COPY_OUT2]] : memref<1x784x32x3xf16, #NWCH, @DDR>) -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK:              }
  //CHECK:              VPURT.Task waits([[BAR0]] : !VPURT.Barrier) {
  //CHECK:                 VPUIP.NNDMA
  //CHECK-SAME:                 inputs([[COPY_IN3]] : memref<1x784x32x3xf16, #NWCH, [@CMX_NN, 3]>) outputs([[COPY_OUT3]] : memref<1x784x32x3xf16, #NWCH, @DDR>) -> memref<1x784x32x3xf16, #NWCH, @DDR>
  //CHECK:              }
  //CHECK:              return [[DDR_OUT]] : memref<1x784x32x6xf16, #NWCH, @DDR>
}
