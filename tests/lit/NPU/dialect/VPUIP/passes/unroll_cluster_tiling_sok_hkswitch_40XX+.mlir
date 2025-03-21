//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-cluster-tiling --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    48x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    48x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x48x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<48x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<48x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x48x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<48x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<48x1x1x4xsi32, #NCHW, @CMX_NN>

//CHECK-LABEL: @UnrollNceSOK
func.func @UnrollNceSOK(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare !WeightsTable_DDR = dense<1> : tensor<48x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <32768> -> !ParentOutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <131072> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <132608> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
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
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
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
                    DPUTask {
                        outStart = [0, 0, 32], outEnd = [31, 31, 47],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 2 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:  [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_TABLE3_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS1_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:  [[WEIGHTS2_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:  [[WEIGHTS3_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:      [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:      [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:      [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:      [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x32x32xf16, #NHWC, @DDR>

    //CHECK:      [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:      [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:      [[IN3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:      [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:      [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x48x32x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:      [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 0, 0, 0]
    //CHECK-SAME:      cluster_id = 0 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 16, 0, 0]
    //CHECK-SAME:      cluster_id = 1 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[OUT3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 32, 0, 0]
    //CHECK-SAME:      cluster_id = 2 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:      [[WEIGHTS1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

    //CHECK:      [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS_TABLE3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>
    //CHECK:      [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS_TABLE3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>


    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weight
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of weight
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS3_CST]] : memref<16x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS3_CMX_COPY]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE3_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE3_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:           output_ITI_buff([[OUT2_CMX]], [[OUT3_CMX]]
    //CHECK:           outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:           variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 32 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN3_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE3_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:           output_ITI_buff([[OUT1_CMX]], [[OUT2_CMX]]
    //CHECK:           outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:        variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 47], outStart = [0, 0, 32],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x48x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x48x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xf16, #NHWC, @CMX_NN, {
    mode = "MULTICASTED|SEGMENTED",
    num_tiles = [1, 1, 3, 1],
    num_clusters = 3,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
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

!Input_DDR = memref<1x16x33x33xf16, #NHWC>
!Output_DDR = memref<1x16x33x33xf16, #NHWC>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

//CHECK-LABEL: @UnrollNceHKSwitch
func.func @UnrollNceHKSwitch(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <11616> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46464> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46976> -> !WeightsTableDistributed

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
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [32, 10, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 11, 0], outEnd = [32, 21, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    DPUTask {
                        outStart = [0, 22, 0], outEnd = [32, 32, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
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

    //CHECK:     [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:     [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:     [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:     [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:     [[IN1_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK:     [[IN2_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <11616> -> memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK:     [[IN3_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <23232> -> memref<1x16x11x33xf16, #NHWC, @DDR>

    //CHECK:     [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x33x33xf16, #NHWC>

    //CHECK:     [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:     [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:     [[IN3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:     [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:     [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:     [[IN3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>


    //CHECK:     [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <11616> -> memref<1x16x33x33xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:     [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, #NHWC, [@CMX_NN, 0]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 22, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 11, 33]
    //CHECK-SAME:      offset = [0, 0, 0, 0]
    //CHECK-SAME:      cluster_id = 0 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:     [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, #NHWC, [@CMX_NN, 1]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 22, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 11, 33]
    //CHECK-SAME:      offset = [0, 0, 11, 0]
    //CHECK-SAME:      cluster_id = 1 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 11, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:     [[OUT3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, #NHWC, [@CMX_NN, 2]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 11, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 11, 33]
    //CHECK-SAME:      offset = [0, 0, 22, 0]
    //CHECK-SAME:      cluster_id = 2 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 22, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 11, 33], offset = [0, 0, 22, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <46464> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <46464> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <46464> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46464> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <46976> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <46976> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <46976> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46976> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN3_DDR]] : memref<1x16x11x33xf16, #NHWC, @DDR>
    //CHECK-SAME:       outputs([[IN3_CMX_COPY]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
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
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:                output_ITI_buff([[OUT2_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:           variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 10, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:            output_ITI_buff([[OUT1_CMX]], [[OUT3_CMX]]
    //CHECK:            outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:        variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 21, 15], outStart = [0, 11, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN3_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE3_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x11x33xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT2_CMX]]
    //CHECK:                outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 32, 15], outStart = [0, 22, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.NNDMA
    // CHECK-SAME:       inputs([[OUT_CMX]] : memref<1x16x33x33xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x33x33xf16, #NHWC>)
    // CHECK:        }

}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x48x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 3, 1, 1],
    num_clusters = 3
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    48x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    48x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [3, 1, 1, 1],
    num_clusters = 3
}>

!Input_DDR = memref<1x16x32x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x48x32x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<48x16x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<48x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x48x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<48x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<48x1x1x4xsi32, #NCHW, @CMX_NN>

//CHECK-LABEL: @UnrollNCEWithDuplicatedAndSegmentedWeights
func.func @UnrollNCEWithDuplicatedAndSegmentedWeights(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare !WeightsTable_DDR = dense<1> : tensor<48x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <32768> -> !ParentOutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <131072> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <132608> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
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
        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
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
                    DPUTask {
                        outStart = [0, 0, 32], outEnd = [31, 31, 47],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 2 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:  [[WEIGHTS_TABLE1_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_TABLE2_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_TABLE3_CST:%.*]] = const.Declare memref<16x1x1x4xsi32, @DDR> = dense<1> : tensor<48x1x1x4xsi32>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_CST:%.*]] = const.Declare memref<48x16x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]

    //CHECK:      [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:      [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:      [[IN_DDR:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x32x32xf16, #NHWC, @DDR>
    //CHECK:      [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x32x32xf16, #NHWC, @DDR>

    //CHECK:      [[IN1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:      [[IN2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:      [[IN3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:      [[IN_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64}>

    //CHECK:      [[OUT_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x48x32x32xf16, #NHWC, [@CMX_NN, 0]>

    //CHECK:      [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 0, 0, 0]
    //CHECK-SAME:      cluster_id = 0 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 16, 0, 0]
    //CHECK-SAME:      cluster_id = 1 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:      ]>

    //CHECK:      [[OUT3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 16, 0, 0], cluster_id = 2 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 16, 32, 32]
    //CHECK-SAME:      offset = [0, 32, 0, 0]
    //CHECK-SAME:      cluster_id = 2 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 32], offset = [0, 32, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <131584> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <132096> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>
    //CHECK:      [[WEIGHTS_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <131072> -> !VPUIP.DistributedBuffer<48x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64}>

    //CHECK:      [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS_TABLE3_CMX:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>
    //CHECK:      [[WEIGHTS_TABLE1_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS_TABLE3_CMX_COPY:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <132608> -> memref<16x1x1x4xsi32, [@CMX_NN, 2]>


    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64}>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<48x16x1x1xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX]] : !VPUIP.DistributedBuffer<48x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [3, 1, 1, 1], num_clusters = 3 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE3_CST]] : memref<16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE3_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:                output_ITI_buff([[OUT2_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 0]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 1]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask {
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 32 : i64, task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       } input([[IN3_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE3_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x32x32xf16, #NHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT2_CMX]]
    //CHECK:                outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, #NHWC, [@CMX_NN, 2]
    //CHECK:        variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 47], outStart = [0, 0, 32],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.*]] : memref<1x48x32x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x48x32x32xf16, #NHWC, @DDR>)
    //CHECK:        }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x432x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x432x16x16xf16, #NCHW, @CMX_NN, {
    mode = "MULTICASTED|SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!Input_DDR = memref<1x432x16x16xf16, #NHWC, @DDR>
!Output_DDR = memref<1x432x16x16xf16, @DDR>

func.func @UnrollNceHKSwitchWithNCHWOutput(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    %network_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %network_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %cmx_in = VPURT.DeclareBuffer <CMX_NN> <110592> -> !InputDistributed
    %cmx_out = VPURT.DeclareBuffer <CMX_NN> <221184> -> !OutputDistributed

    %bar_dma_in = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar_nce = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar_dma_in : !VPURT.Barrier) {
      %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%network_in : !Input_DDR) outputs(%cmx_in : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%bar_dma_in : !VPURT.Barrier) updates(%bar_nce : !VPURT.Barrier) {
      %10 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 4699 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%cmx_in : !InputDistributed) weights(%cmx_in : !InputDistributed) parent_input(%cmx_in : !InputDistributed) parent_output(%cmx_out : !OutputDistributed) outputs(%cmx_out : !OutputDistributed) -> !OutputDistributed  variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 7, 431], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 431], outStart = [0, 8, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar_nce : !VPURT.Barrier) {
      %10 = VPUIP.NNDMA {port = 0 : i64} inputs(%cmx_out : !OutputDistributed) outputs(%network_out : !Output_DDR) -> !Output_DDR
    }
    return %output : !Output_DDR

    // CHECK-DAG: [[IN_DDR_0:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x432x8x16xf16, #NHWC, @DDR>
    // CHECK-DAG: [[IN_DDR_1:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, @DDR>
    // CHECK-DAG: [[OUT_DDR:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x432x16x16xf16, @DDR>

    // CHECK-DAG: [[WEIGHTS_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[WEIGHTS_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>
    // CHECK-DAG: [[NCE_INPUT_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[NCE_INPUT_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK-DAG: [[DDR_TO_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG: [[DDR_TO_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <110592> -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>

    // CHECK-DAG: [[OUT_CMX_TO_DDR:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <221184> -> memref<1x432x16x16xf16, [@CMX_NN, 0]>

    // CHECK-DAG: [[OUT_CMX_0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <221184> -> !VPUIP.ITIBuffer<
    // CHECK-NEXT: 1x432x16x16xf16, #NCHW, [@CMX_NN, 0],
    // CHECK-NEXT:       inwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:         ],
    // CHECK-NEXT:       outwardHaloRegions = [
    // CHECK-NEXT:           #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ]>
    // CHECK-NEXT:      ]>
    // CHECK-DAG: [[OUT_CMX_1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <221184> -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:    1x432x16x16xf16, #NCHW, [@CMX_NN, 1],
    // CHECK-NEXT:    inwardHaloRegions = [
    // CHECK-NEXT:       #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:    ],
    // CHECK-NEXT:    outwardHaloRegions = [
    // CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:      ]>
    // CHECK-NEXT:   ]>

    // CHECK: [[BAR_DMA:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK: [[BAR_NCE:%.*]]  = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK: VPURT.Task updates([[BAR_DMA]] : !VPURT.Barrier) {
    // CHECK:  VPUIP.NNDMA
    // CHECK-SAMEL inputs([[IN_DDR_0]] : memref<1x432x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[DDR_TO_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK: VPURT.Task updates([[BAR_DMA]] : !VPURT.Barrier) {
    // CHECK:    VPUIP.NNDMA
    // CHECK-SAME: inputs([[IN_DDR_1]] : memref<1x432x8x16xf16, #NHWC, @DDR>)
    // CHECK-SAME: outputs([[DDR_TO_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>) -> memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>


    // CHECK: VPURT.Task waits([[BAR_DMA]] : !VPURT.Barrier) updates([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK: VPUIP.NCEClusterTask
    // CHECK-SAME:  {task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:  input([[NCE_INPUT_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[WEIGHTS_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[NCE_INPUT_CMX_0]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_output([[OUT_CMX_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:   1x432x16x16xf16, #NCHW, [@CMX_NN, 0],
    // CHECK-NEXT:    inwardHaloRegions = [
    // CHECK-NEXT:      #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:    ],
    // CHECK-NEXT:    outwardHaloRegions = [
    // CHECK-NEXT:     #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:     ]>
    // CHECK-NEXT:  ]>)
    // CHECK-SAME:  output_ITI_buff([[OUT_CMX_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:      1x432x16x16xf16, #NCHW, [@CMX_NN, 1],
    // CHECK-NEXT:      inwardHaloRegions = [
    // CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:     ],
    // CHECK-NEXT:      outwardHaloRegions = [
    // CHECK-NEXT:        #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:         ]>
    // CHECK-NEXT:      ]>)
    // CHECK-SAME:  outputs([[OUT_CMX_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:                          1x432x16x16xf16, #NCHW, [@CMX_NN, 0],
    // CHECK-NEXT:                          inwardHaloRegions = [
    // CHECK-NEXT:                              #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:                          ],
    // CHECK-NEXT:                           outwardHaloRegions = [
    // CHECK-NEXT:                             #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                                 #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:                              ]>
     // CHECK-NEXT:                         ]>)
    // CHECK-SAME:  -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:     1x432x16x16xf16, #NCHW, [@CMX_NN, 0],
    // CHECK-NEXT:     inwardHaloRegions = [
    // CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:     ],
    // CHECK-NEXT:     outwardHaloRegions = [
    // CHECK-NEXT:        #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:        ]>
    // CHECK-NEXT:    ]>
    // CHECK:    variants : {
    // CHECK:      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 7, 431], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:    } PPE : {
    // CHECK:      PPETask {ppe = #VPU.PPEStub<>}

    // CHECK: VPURT.Task waits([[BAR_DMA]] : !VPURT.Barrier) updates([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK:  VPUIP.NCEClusterTask
    // CHECK-SAME:  {task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:  input([[NCE_INPUT_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:  weights([[WEIGHTS_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK-SAME:  parent_input([[NCE_INPUT_CMX_1]] : memref<1x432x8x16xf16, #NHWC, [@CMX_NN, 1]>)
    // CHECK:  parent_output([[OUT_CMX_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:                           1x432x16x16xf16, #NCHW, [@CMX_NN, 1],
    // CHECK-NEXT:                           inwardHaloRegions = [
    // CHECK-NEXT:                              #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:                           ],
    // CHECK-NEXT:                           outwardHaloRegions = [
    // CHECK-NEXT:                               #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                                   #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:                               ]>
    // CHECK-NEXT:                           ]>)
    // CHECK:  output_ITI_buff([[OUT_CMX_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:                              1x432x16x16xf16, #NCHW, [@CMX_NN, 0],
    // CHECK-NEXT:                              inwardHaloRegions = [
    // CHECK-NEXT:                                  #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:                              ],
    // CHECK-NEXT:                              outwardHaloRegions = [
    // CHECK-NEXT:                                 #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                                    #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:                                 ]>
    // CHECK-NEXT:                             ]>)
    // CHECK:  outputs([[OUT_CMX_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:                      1x432x16x16xf16, #NCHW, [@CMX_NN, 1],
    // CHECK-NEXT:                      inwardHaloRegions = [
    // CHECK-NEXT:                         #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:                      ],
    // CHECK-NEXT:                     outwardHaloRegions = [
    // CHECK-NEXT:                         #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                           #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:                         ]>
    // CHECK-NEXT:                    ]>) -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:                           1x432x16x16xf16, #NCHW, [@CMX_NN, 1],
    // CHECK-NEXT:                           inwardHaloRegions = [
    // CHECK-NEXT:                              #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:                           ],
    // CHECK-NEXT:                           outwardHaloRegions = [
    // CHECK-NEXT:                              #VPUIP.OutwardHaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                                  #VPUIP.HaloRegionAttr<shape = [1, 432, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:                              ]>
    // CHECK-NEXT:                           ]>
    // CHECK: variants : {
    // CHECK:    DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 431], outStart = [0, 8, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:    } PPE : {
    // CHECK:      PPETask {ppe = #VPU.PPEStub<>}

    // CHECK: VPURT.Task waits([[BAR_NCE]] : !VPURT.Barrier) {
    // CHECK:   VPUIP.NNDMA
    // CHECK:   inputs([[OUT_CMX_TO_DDR]] : memref<1x432x16x16xf16, [@CMX_NN, 0]>)
    // CHECK:   outputs([[OUT_DDR]] : memref<1x432x16x16xf16, @DDR>) -> memref<1x432x16x16xf16, @DDR>

    // CHECK: return %arg1 : memref<1x432x16x16xf16, @DDR>
}
