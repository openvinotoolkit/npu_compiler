//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed_1 = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentOutputDistributed_1 = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed_1 = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsTableDistributed_1 = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!SpilledInputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentInputDistributed_2 = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentOutputDistributed_2 = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed_2 = !VPUIP.DistributedBuffer<
    32x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsTableDistributed_2 = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!SpilledOutput_DDR = memref<1x32x32x32xf16, #NHWC, @DDR>

!Weights1_DDR = memref<32x16x1x1xf16, #NHWC, @DDR>
!WeightsTable1_DDR = memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub1_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub1_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub1_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub1_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

!Weights2_DDR = memref<32x32x1x1xf16, #NHWC, @DDR>
!WeightsTable2_DDR = memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub2_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!OutputStub2_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub2_CMX = memref<32x32x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub2_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights1_cst = const.Declare !Weights1_DDR =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table1_cst = const.Declare !WeightsTable1_DDR = dense<1> : tensor<32x1x1x4xsi32>

    %weights2_cst = const.Declare !Weights2_DDR =
        dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table2_cst = const.Declare !WeightsTable2_DDR = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR
    %spilled_out = VPURT.DeclareBuffer <DDR> <0> -> !SpilledOutput_DDR

    // CMX buffers/ 1st task
    %parent_input1_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed_1
    %weights1 = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed_1
    %weights_table1 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed_1
    %parent_out1_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed_1

    // CMX buffers/ 2nd task
    %spilled_input_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !SpilledInputDistributed
    %parent_input2_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentInputDistributed_2
    %weights2 = VPURT.DeclareBuffer <CMX_NN> <99072> -> !WeightsDistributed_2
    %weights_table2 = VPURT.DeclareBuffer <CMX_NN> <99584> -> !WeightsTableDistributed_2
    %parent_out2_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentOutputDistributed_2

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input1_cmx: !ParentInputDistributed_1) -> !ParentInputDistributed_1
    }

    // Upload weights/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights1_cst: !Weights1_DDR) outputs(%weights1: !WeightsDistributed_1) -> !WeightsDistributed_1
    }

    // Upload weights table/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table1_cst: !WeightsTable1_DDR) outputs(%weights_table1: !WeightsTableDistributed_1) -> !WeightsTableDistributed_1
    }

    // Cluster tiling/ 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input1_cmx : !ParentInputDistributed_1)
                    weights(%weights1 : !WeightsDistributed_1)
                    weight_table(%weights_table1 : !WeightsTableDistributed_1)
                    parent_input(%parent_input1_cmx : !ParentInputDistributed_1)
                    parent_output(%parent_out1_cmx : !ParentOutputDistributed_1)
                    outputs(%parent_out1_cmx : !ParentOutputDistributed_1)
                        -> !ParentOutputDistributed_1 variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Spill 1st output
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out1_cmx: !ParentOutputDistributed_1) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
    }

    // Upload input
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%spilled_out: !SpilledOutput_DDR) outputs(%spilled_input_cmx: !SpilledInputDistributed) -> !SpilledInputDistributed
    }

    // Upload weights/ 2nd task
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights2_cst: !Weights2_DDR) outputs(%weights2: !WeightsDistributed_2) -> !WeightsDistributed_2
    }

    // Upload weights table/ 2nd task
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%weights_table2_cst: !WeightsTable2_DDR) outputs(%weights_table2: !WeightsTableDistributed_2) -> !WeightsTableDistributed_2
    }

    // Cluster tiling/ 2nd task
    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input2_cmx : !ParentInputDistributed_2)
                    weights(%weights2 : !WeightsDistributed_2)
                    weight_table(%weights_table2 : !WeightsTableDistributed_2)
                    parent_input(%parent_input2_cmx : !ParentInputDistributed_2)
                    parent_output(%parent_out2_cmx : !ParentOutputDistributed_2)
                    outputs(%parent_out2_cmx : !ParentOutputDistributed_2)
                        -> !ParentOutputDistributed_2 variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar4: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out2_cmx: !ParentOutputDistributed_2) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST_2ND_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST_2ND_TASK:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR>

    //CHECK-DAG:    [[WEIGHTS1_CST_2ND_TASK:%.*]] = const.Declare memref<16x32x1x1xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[WEIGHTS2_CST_2ND_TASK:%.*]] = const.Declare memref<16x32x1x1xf16, #NHWC, @DDR>

    //CHECK:        [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR4:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[SPILLED_OUTPUT_DDR:%.*]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x32x32x32xf16, #NHWC, @DDR>

    //CHECK-DAG:    [[SPILLED_OUTPUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:        [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK-DAG:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    // Upload 1st part of weights/ 2nd task
    //CHECK-DAG:    [[WEIGHTS1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // Upload 2nd part of weights/ 2nd task
    //CHECK-DAG:    [[WEIGHTS2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99584> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99584> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>

    // Upload 1st part of weights table/ 2nd task
    //CHECK:    [[WEIGHTS_TABLE1_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <99584> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    // Upload 2nd part of weights table/ 2nd task
    //CHECK:    [[WEIGHTS_TABLE2_CMX_COPY_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <99584> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>


    //CHECK:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x32x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:    ]
    //CHECK:        outwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 0, 0, 0]
    //CHECK-SAME:      cluster_id = 0 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:           #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x32x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:    ]
    //CHECK-NEXT:     outwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 16, 0, 0]
    //CHECK-SAME:      cluster_id = 1 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:           #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:       ]>

    // Upload input, weights, weights table for 1st cluster task
    // Process 1st cluster task

    // Spill 1st output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[SPILLED_OUTPUT_CMX]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[SPILLED_OUTPUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Upload input
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[SPILLED_OUTPUT_DDR]] : memref<1x32x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[INPUT2_CMX_COPY]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK:        }

   // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:                parent_output([[OUT1_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:                outputs([[OUT1_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK-:           variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<16x32x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:                parent_output([[OUT2_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:                outputs([[OUT2_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }
}
