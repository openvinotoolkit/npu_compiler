//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x1x16x1x4xf16, #GNHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x1x32x1x4xf16, #GNHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    1x32x16x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    1x32x1x1x4xsi32, #NCDHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x1x16x1x4xf16, {order = #GNHWC}, @DDR>
!Output_DDR = memref<1x1x32x1x4xf16, {order = #GNHWC}, @DDR>
!Weights_DDR = memref<1x32x16x1x1xf16, {order = #GNHWC}, @DDR>
!WeightsTable_DDR = memref<1x32x1x1x4xsi32, @DDR>

// MatMul (NCEClusterTask with task_type CONV and 5-d GNHWC operands) with the SplitOverKernel
// strategy: weights/weight-table are SEGMENTED over the OC dim, input is DUPLICATED, and the
// output is SEGMENTED|DUPLICATED, broadcast to the other cluster via ITI halo regions.
//CHECK-LABEL: @UnrollMatMulSOK
func.func @UnrollMatMulSOK(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<1x32x16x1x1xf16>, [#const.Reorder<#GNHWC>]
    %weights_table_cst = const.Declare !WeightsTable_DDR = dense<1> : tensor<1x32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <2048> -> !ParentOutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <4096> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <8192> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input_cmx : !ParentInputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !ParentInputDistributed)
                    parent_output(%parent_out_cmx : !ParentOutputDistributed)
                    outputs(%parent_out_cmx : !ParentOutputDistributed)
                        -> !ParentOutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [3, 0, 15],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [3, 0, 31],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:  [[WEIGHTS_TABLE1_CST:%.+]] = const.Declare memref<1x16x1x1x4xsi32, @DDR> = dense<1> : tensor<1x32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0, 0], [1, 16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS_TABLE2_CST:%.+]] = const.Declare memref<1x16x1x1x4xsi32, @DDR> = dense<1> : tensor<1x32x1x1x4xsi32>, [#const.SubView<[0, 16, 0, 0, 0], [1, 16, 1, 1, 4]>]
    //CHECK-DAG:  [[WEIGHTS1_CST:%.+]] = const.Declare memref<1x16x16x1x1xf16, {order = #GNHWC}, @DDR> = dense<1.000000e+00> : tensor<1x32x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0, 0], [1, 16, 16, 1, 1]>, #const.Reorder<#GNHWC>]
    //CHECK-DAG:  [[WEIGHTS2_CST:%.+]] = const.Declare memref<1x16x16x1x1xf16, {order = #GNHWC}, @DDR> = dense<1.000000e+00> : tensor<1x32x16x1x1xf16>, [#const.SubView<[0, 16, 0, 0, 0], [1, 16, 16, 1, 1]>, #const.Reorder<#GNHWC>]

    //CHECK:      [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:      [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:      [[IN_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x1x16x1x4xf16, {order = #GNHWC}, @DDR>
    //CHECK:      [[OUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x1x32x1x4xf16, {order = #GNHWC}, @DDR>

    //CHECK:      [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    //CHECK:      [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]>
    //CHECK:      [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x1x16x1x4xf16, #GNHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:      [[OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>

    //CHECK:      [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 1, 16, 1, 4], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 1, 16, 1, 4]
    //CHECK-SAME:      offset = [0, 0, 0, 0, 0]
    //CHECK-SAME:      cluster_id = 0 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 1, 16, 1, 4], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]
    //CHECK-NEXT:   inwardHaloRegions = [
    //CHECK-NEXT:     #VPUIP.HaloRegionAttr<shape = [1, 1, 16, 1, 4], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
    //CHECK-NEXT:   ]
    //CHECK-NEXT:   outwardHaloRegions = [
    //CHECK-NEXT:       #VPUIP.OutwardHaloRegionAttr<
    //CHECK-SAME:      shape = [1, 1, 16, 1, 4]
    //CHECK-SAME:      offset = [0, 0, 16, 0, 0]
    //CHECK-SAME:      cluster_id = 1 : i64
    //CHECK-SAME:      inwardHaloRegions = [
    //CHECK-NEXT:        #VPUIP.HaloRegionAttr<shape = [1, 1, 16, 1, 4], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:       ]>

    //CHECK:      [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <4096> -> memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 1]>

    //CHECK:      [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <8192> -> memref<1x16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> memref<1x16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <8192> -> memref<1x16x1x1x4xsi32, [@CMX_NN, 1]>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x1x16x1x4xf16, {order = #GNHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x1x16x1x4xf16, #GNHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<1x16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<1x16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<1x16x1x1x4xsi32, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<1x16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }

    // 1st task (MatMul on cluster 0)
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<1x16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]
    //CHECK:           output_ITI_buff([[OUT2_CMX]]
    //CHECK:           outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]
    //CHECK:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 0, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 2nd task (MatMul on cluster 1)
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<1x16x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<1x16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x1x16x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]
    //CHECK:                output_ITI_buff([[OUT1_CMX]]
    //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 1]
    //CHECK:           variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 0, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.+]] : memref<1x1x32x1x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x1x32x1x4xf16, {order = #GNHWC}, @DDR>)
    //CHECK:        }
}
