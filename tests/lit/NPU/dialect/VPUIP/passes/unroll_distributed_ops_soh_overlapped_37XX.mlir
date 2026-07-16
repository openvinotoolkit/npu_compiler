//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// In this example DPU task in SOHoverlapped mode with input 1x16x32x32
// will be split into 2 clusters where each input will be 1x16x17x32
!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x30x30xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x30x30xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, {order = #NHWC}, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x32x32xf16, @CMX_NN>
!OutputStub_CMX = memref<1x16x30x30xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @UnrollNCE
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x32x32xf16, {order = #NHWC}, @DDR>, [[ARG_1:%.+]]: memref<1x16x30x30xf16, {order = #NHWC}, @DDR>)
func.func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
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
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [29, 14, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 15, 0], outEnd = [29, 29, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.+]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x3x3xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <15360> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x15x30xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <14400> -> memref<1x16x15x30xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[PARENT_IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <31808> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <31808> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>



    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [29, 14, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x30x30xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [29, 29, 15], outStart = [0, 15, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x15x30xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x15x30xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x15x30xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]] : memref<1x16x30x30xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [5, 5],
    pads = #VPU.Padding<left = 2 , right = 2, top = 2, bottom = 2>,
    strides = [1, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, {order = #NHWC}, @DDR>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @UnrollNCEWithInconsistentPadAndKernelSize
// CHECK-SAME:      [[ARG_0:%[^:]+]]: memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
func.func @UnrollNCEWithInconsistentPadAndKernelSize(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
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
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                    kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [31, 15, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [31, 31, 15], outStart = [0, 16, 0],
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.+]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x3x3xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x18x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <14336> -> memref<1x16x18x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <16384> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[PARENT_IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [5, 5], pads = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <31808> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <31808> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <31808> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34112> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34112> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x18x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x18x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [5, 5], pads = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 15, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x18x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [5, 5], pads = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, strides = [1, 1], num_clusters = 2 : i64}>
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 16, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<1x16x112x112xi1,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [1, 1], pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<1x16x112x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x112x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
func.func @AdjustSizeAlignmentk1x1(%output: !DistBuf) -> !DistBuf {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !DistBuf
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !DistBuf

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !DistBuf) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !DistBuf) -> !DistBuf
    }
    return %18 : !DistBuf

    // Check alignment is set back to 512 for per cluster buffers
    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <12800> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <12800> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistBuf = !VPUIP.DistributedBuffer<1x16x112x112xi1,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [5, 5], pads = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>, strides = [1, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<1x16x112x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x112x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
func.func @AdjustSizeAlignmentk5x5(%output: !DistBuf) -> !DistBuf {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !DistBuf
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !DistBuf

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !DistBuf) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !DistBuf) -> !DistBuf
    }
    return %18 : !DistBuf

    // Check alignment is set back to 512 for per cluster buffers
    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <13312> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <13312> {swizzlingKey = 5 : i64} -> memref<1x16x58x112xi1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}


// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
!DistBuf  = !VPUIP.DistributedBuffer<1x2x4x3xf16, {order = #NCWH, strides = [192, 12, 1, 4]}, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 2, 2, 3], [1, 2, 2, 3]], compute_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]], memory_shapes = [[1, 2, 2, 3], [1, 2, 2, 3]], memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]}>

func.func @UnrollOverLappedUpdateStride() -> memref<1x2x4x3xf16, {order = #NCWH}, @DDR> {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !DistBuf
    %1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x4x3xf16, {order = #NCWH}, @DDR>

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %20 = VPUIP.NNDMA inputs(%0 : !DistBuf) outputs(%1 : memref<1x2x4x3xf16, {order = #NCWH}, @DDR>) -> memref<1x2x4x3xf16, {order = #NCWH}, @DDR>
    }
    return %1 : memref<1x2x4x3xf16, {order = #NCWH}, @DDR>

    //CHECK:        [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[CMXBUFF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x2x2x3xf16, {order = #NCWH, strides = [96, 6, 1, 2]}, [@CMX_NN, 0]>
    //CHECK:        [[CMXBUFF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x2x2x3xf16, {order = #NCWH, strides = [96, 6, 1, 2]}, [@CMX_NN, 1]>
    //CHECK:        [[DDRBUFF_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x4x3xf16, {order = #NCWH}, @DDR>
    //CHECK:        [[DDRBUFF_1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>
    //CHECK:        [[DDRBUFF_2:%.+]] = VPURT.DeclareBuffer <DDR> <4> -> memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>
    //CHECK:        VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:          [[DMA0:%.+]] = VPUIP.NNDMA <{port = 0 : i64}> inputs([[CMXBUFF_0]] : memref<1x2x2x3xf16, {order = #NCWH, strides = [96, 6, 1, 2]}, [@CMX_NN, 0]>) outputs([[DDRBUFF_1]] : memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>) -> memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR_0]] : !VPURT.Barrier) updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:          [[DMA2:%.+]] = VPUIP.NNDMA <{port = 1 : i64}> inputs([[CMXBUFF_1]] : memref<1x2x2x3xf16, {order = #NCWH, strides = [96, 6, 1, 2]}, [@CMX_NN, 1]>) outputs([[DDRBUFF_2]] : memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>) -> memref<1x2x2x3xf16, {order = #NCWH, strides = [24, 12, 1, 4]}, @DDR>
    //CHECK:        }
    //CHECK:        return [[DDRBUFF_0]] : memref<1x2x4x3xf16, {order = #NCWH}, @DDR>
}
