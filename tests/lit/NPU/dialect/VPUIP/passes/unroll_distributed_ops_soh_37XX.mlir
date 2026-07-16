//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, {order = #NHWC}>

!InputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>

!Buffer0_CMX = memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
!Buffer1_CMX = memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

//CHECK-LABEL: @UnrollNNDMA
func.func @UnrollNNDMA(%input: memref<1x16x33x32xf16>, %output: memref<1x16x33x32xf16>) -> memref<1x16x33x32xf16> {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <DDR> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <DDR> <33792> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Buffer0_CMX
    %input2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Buffer1_CMX

    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> !Buffer0_CMX
    %output2 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> !Buffer1_CMX

    %weights = VPURT.DeclareBuffer <CMX_NN> <34816> -> !WeightsDistributed

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteDMA {dst_order = #NHWC} <{mem_perm = #NHWC}>
            inputs(%input: memref<1x16x33x32xf16>)
            outputs(%parent_in: !Input_DDR)
            -> !Input_DDR
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload input
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }

    // Simulate 1st task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%input1: !Buffer0_CMX) outputs(%output1: !Buffer0_CMX) -> !Buffer0_CMX
    }

    // Simulate 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%input2: !Buffer1_CMX) outputs(%output2: !Buffer1_CMX) -> !Buffer1_CMX
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    // Reorder output

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
        VPUIP.PermuteDMA {dst_order = #NCHW} <{mem_perm = #NCHW}>
            inputs(%parent_out: !Output_DDR)
            outputs(%output: memref<1x16x33x32xf16>)
            -> memref<1x16x33x32xf16>
    }

    return %output: memref<1x16x33x32xf16>

    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <33792> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <51200> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Simulate tasks
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>
!Output_CMX = memref<1x16x33x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
!InputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>

//CHECK-LABEL: @UnrollNNDMACMX2CMX
func.func @UnrollNNDMACMX2CMX() -> !Output_CMX {
    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer <CMX_NN> [0] <33792> -> !Output_CMX
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%input: !InputDistributed) outputs(%output: !Output_CMX) -> !Output_CMX
    }

    return %output: !Output_CMX


    //CHECK: [[BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: [[BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK: [[OUTBUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33792> -> memref<1x16x33x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: [[OUTBUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33792> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: [[OUTBUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <51200> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:    VPUIP.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs([[OUTBUF_0]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: }
    //CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
    //CHECK:    VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF_1]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs([[OUTBUF_1]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK: }
    //CHECK: return [[OUTBUF]] : memref<1x16x33x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ProfilingDistributed = !VPUIP.DistributedBuffer<
    4xui64, affine_map<(d0) -> (d0)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2],
    num_clusters = 2 : i64
}>

!Input_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, {order = #NHWC}>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

!Profiling_CMX = memref<4xui64, [@CMX_NN, 0]>
!Profiling_DDR = memref<4xui64>

// CHECK-LABEL: @UnrollNCEWithProfiling
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>, [[ARG_1:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>, [[ARG_2:%.+]]: memref<4xui64>)
func.func @UnrollNCEWithProfiling(%input: !Input_DDR, %output: !Output_DDR, %prof_output: !Profiling_DDR) -> (!Output_DDR, !Profiling_DDR) {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR
    %profiling_out = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> !Profiling_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !WeightsTableDistributed

    // Profiling buffers
    %prof_buffer_cmx = VPURT.DeclareBuffer <CMX_NN> <353584> -> !ProfilingDistributed

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
        %1:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 1, 0, 0, 0>} <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>,
                    profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 1 : i64>
                }>  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx : !InputDistributed)
                    parent_output(%parent_out_cmx : !OutputDistributed)
                    outputs(%parent_out_cmx : !OutputDistributed)
                    profiling_data(%prof_buffer_cmx: !ProfilingDistributed)
                        -> !OutputDistributed, !ProfilingDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 16, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 17, 0], outEnd = [31, 32, 31],
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

    // Copyback profiling
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%prof_buffer_cmx: !ProfilingDistributed) outputs(%profiling_out: !Profiling_DDR) -> !Profiling_DDR
    }

    return %output, %prof_output: !Output_DDR, !Profiling_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.+]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[PROF1_DDR:%.+]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64, @DDR>
    //CHECK:    [[PROF2_DDR:%.+]]  = VPURT.DeclareBuffer <ProfilingOutput> [0] <16> -> memref<2xui64, @DDR>


    //CHECK:    [[PARENT_IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[PROF1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <353584> -> memref<2xui64, [@CMX_NN, 0]>
    //CHECK:    [[PROF2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <353584> -> memref<2xui64, [@CMX_NN, 1]>
    //CHECK:    [[PROF1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <353584> -> memref<2xui64, [@CMX_NN, 0]>
    //CHECK:    [[PROF2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <353584> -> memref<2xui64, [@CMX_NN, 1]>


    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 0 : i64>,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           profiling_data([[PROF1_CMX]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 1 : i64, numVariants = 1 : i64, clusterId = 1 : i64>,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           profiling_data([[PROF2_CMX]] : memref<2xui64, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[PROF1_CMX_COPY]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[PROF1_DDR]] : memref<2xui64, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[PROF2_CMX_COPY]] : memref<2xui64, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[PROF2_DDR]] : memref<2xui64, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]], [[ARG_2]] : memref<1x16x33x32xf16, {order = #NHWC}, @DDR>, memref<4xui64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
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

!Input_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, {order = #NHWC}>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @UnrollNCESegmentedConv
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>, [[ARG_1:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>)
func.func @UnrollNCESegmentedConv(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx    = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <39424> -> !WeightsTableDistributed

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
                        outStart = [0, 0, 0], outEnd = [31, 16, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 17, 0], outEnd = [31, 32, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
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
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x3x3xf16, {order = #NHWC}>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>

    //CHECK-DAG:    [[PARENT_IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK-DAG:    [[PARENT_OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK-DAG:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <39424> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, {order = #NHWC}>)
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
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]] : memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, {order = #NHWC}>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @UnrollNCESequence
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>, [[ARG_1:%.+]]: memref<1x16x33x32xf16, {order = #NHWC}, @DDR>)
func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !WeightsTableDistributed

    %parent_input_cmx_2 = VPURT.DeclareBuffer <CMX_NN> <17408> -> !InputDistributed
    %parent_out_cmx_2 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed

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
                    kernel_size = [1, 1],
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
                        outStart = [0, 0, 0], outEnd = [31, 16, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 17, 0], outEnd = [31, 32, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Cluster tiling
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input_cmx_2 : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table(%weights_table : !WeightsTableDistributed)
                    parent_input(%parent_input_cmx_2 : !InputDistributed)
                    parent_output(%parent_out_cmx_2 : !OutputDistributed)
                    outputs(%parent_out_cmx_2 : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 16, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 17, 0], outEnd = [31, 32, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx_2: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.+]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}>

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[PARENT_IN_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    [[PARENT_OUT_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[PARENT_IN_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }


    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:       input([[IN2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]] : memref<1x16x33x32xf16, {order = #NHWC}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!typeCmxDistributed = !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!type_CMX_memref = memref<1x4x512x1xf16, @CMX_NN>


!Input_DDR  = memref<1x4x512x1xf16, @DDR>
!Output_DDR = memref<1x4x512x1xf16, @DDR>


VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func nested @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollTanhSOH
func.func @UnrollTanhSOH(%input0: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %395 = VPURT.DeclareBuffer <DDR> <4096> -> !Input_DDR
    %302 = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR

    %300 = VPURT.DeclareBuffer <CMX_NN> <0> -> !typeCmxDistributed
    %301 = VPURT.DeclareBuffer <CMX_NN> <2048> -> !typeCmxDistributed


    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %399 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%395 : !Input_DDR) outputs(%300 : !typeCmxDistributed) -> !typeCmxDistributed
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
         %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
         @VPU.SW::@builtin_TanhOp inputs(%300 as %arg4: !typeCmxDistributed) outputs(%301 as %arg5: !typeCmxDistributed) on tile 0 -> !typeCmxDistributed  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : !typeCmxDistributed, !typeCmxDistributed
         }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  attributes {isTrailingSWLayer = false} {
        %399 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%301 : !typeCmxDistributed) outputs(%302 : !Output_DDR) -> !Output_DDR
    }


    return %output: !Output_DDR

    //CHECK:    [[IN_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <4096> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[IN_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <4608> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[OUT_DDR1:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[OUT_DDR2:%.+]] = VPURT.DeclareBuffer <DDR> <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[IN0_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[IN0_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUT0_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUT0_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.NNDMA <{port = 0 : i64}> inputs([[IN_DDR1]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs([[IN0_CMX_COPY]] : memref<1x4x256x1xf16, [@CMX_NN, 0]>) -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    VPUIP.NNDMA <{port = 1 : i64}> inputs([[IN_DDR2]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs([[IN1_CMX_COPY]] : memref<1x4x256x1xf16, [@CMX_NN, 1]>) -> memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs([[IN0_CMX]] as [[ARG2:%.+]]: memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs([[OUT0_CMX_COPY]] as [[ARG3:%.+]]: memref<1x4x256x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x256x1xf16, [@CMX_NN, 0]>{
    //CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[ARG2]], [[ARG3]]) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_TanhOp inputs([[IN1_CMX]] as [[ARG22:%.+]]: memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs([[OUT1_CMX_COPY]] as [[ARG23:%.+]]: memref<1x4x256x1xf16, [@CMX_NN, 1]>) on tile 1 -> memref<1x4x256x1xf16, [@CMX_NN, 1]>{
    //CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[ARG22]], [[ARG23]]) : memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.NNDMA <{port = 0 : i64}> inputs([[OUT0_CMX]] : memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs([[OUT_DDR1]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    VPUIP.NNDMA <{port = 1 : i64}> inputs([[OUT1_CMX]] : memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs([[OUT_DDR2]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!typeCmxDistributedSOH = !VPUIP.DistributedBuffer<
    1x1x666x400xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!typeDdrDistributedSOH = !VPUIP.DistributedBuffer<
    1x1x666x400xf32, #NCHW, @DDR, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!typeCmxDistributedOutSOH = !VPUIP.DistributedBuffer<
    1x1x666x400xui8, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!typeCmxDuplicated = !VPUIP.DistributedBuffer<
    1x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!typeCmxOutDuplicated = !VPUIP.DistributedBuffer<
    1x1x1x1xui8, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func nested @builtin_DynamicQuantize(memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, memref<*xf32, @CMX_NN>, memref<*xui8, @CMX_NN>, memref<*xf32, @CMX_NN>, memref<*xui8, @CMX_NN>) attributes {VPU.kernel_code = "dynamic_quantize.cpp", VPU.kernel_entry = "dynamic_quantize", VPU.kernel_name = "dynamic_quantize", VPU.task_type = @COMPUTE}
    func.func nested @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollDynamicQuantizeSOH
func.func @UnrollDynamicQuantizeSOH() -> (memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>, memref<1x1x1x1xf32, @DDR>, memref<1x1x1x1xui8, @DDR>) {
    %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <60928> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> <60992> -> !typeCmxDuplicated

    %6 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xf32, @DDR>
    %7 = VPURT.DeclareBuffer <CMX_NN> <128> -> !typeCmxDuplicated

    %8 = VPURT.DeclareBuffer <NetworkInput> [0] <7468800> -> memref<1x1x666x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <CMX_NN> <194624> -> !typeCmxDistributedSOH

    %10 = VPURT.DeclareBuffer <CMX_NN> <995904> -> !typeCmxDistributedOutSOH
    %11 = VPURT.DeclareBuffer <CMX_NN> <727424> -> !typeCmxDuplicated
    %12 = VPURT.DeclareBuffer <CMX_NN> <727488> -> !typeCmxOutDuplicated

    %13 = VPURT.DeclareBuffer <NetworkOutput> [0] <1334000> -> memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    %14 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x1x1x1xf32, @DDR>
    %15 = VPURT.DeclareBuffer <NetworkOutput> [2] <0> -> memref<1x1x1x1xui8, @DDR>

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%4 : memref<1x1x1x1xf32, [@CMX_NN, 0]>) outputs(%5 : !typeCmxDuplicated) -> !typeCmxDuplicated
    }

    VPURT.Task updates(%1 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%6 : memref<1x1x1x1xf32, @DDR>) outputs(%7 : !typeCmxDuplicated) -> !typeCmxDuplicated
    }

    VPURT.Task updates(%2 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%8 : memref<1x1x666x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) outputs(%9 : !typeCmxDistributedSOH) -> !typeCmxDistributedSOH
    }

    VPURT.Task waits(%0, %1, %2 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%3 : !VPURT.Barrier) {
      %results:3 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>} @VPU.SW::@builtin_DynamicQuantize inputs(%9 as %arg4: memref<1x1x666x400xf32, @CMX_NN>, %5 as %arg5: memref<1x1x1x1xf32, @CMX_NN>, %7 as %arg6: memref<1x1x1x1xf32, @CMX_NN>) outputs(%10 as %arg7: memref<1x1x666x400xui8, @CMX_NN>, %11 as %arg8: memref<1x1x1x1xf32, @CMX_NN>, %12 as %arg9: memref<1x1x1x1xui8, @CMX_NN>) on tile 0 -> (!typeCmxDistributedOutSOH, !typeCmxDuplicated, !typeCmxOutDuplicated){
        VPUIP.SW.Kernel.run(%arg4, %arg5, %arg6, %arg7, %arg8, %arg9) : memref<1x1x666x400xf32, @CMX_NN>, memref<1x1x1x1xf32, @CMX_NN>, memref<1x1x1x1xf32, @CMX_NN>, memref<1x1x666x400xui8, @CMX_NN>, memref<1x1x1x1xf32, @CMX_NN>, memref<1x1x1x1xui8, @CMX_NN>
      }
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%10 : !typeCmxDistributedOutSOH) outputs(%13 : memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) -> memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    }

    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%11 : !typeCmxDuplicated) outputs(%14 : memref<1x1x1x1xf32, @DDR>) -> memref<1x1x1x1xf32, @DDR>
    }
    VPURT.Task waits(%3 : !VPURT.Barrier) {
      %2631 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%12 : !typeCmxOutDuplicated) outputs(%15 : memref<1x1x1x1xui8, @DDR>) -> memref<1x1x1x1xui8, @DDR>
    }

    return %13, %14, %15 : memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>, memref<1x1x1x1xf32, @DDR>, memref<1x1x1x1xui8, @DDR>

    // CHECK: [[MIN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <60928> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK: [[CMX_0_MIN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <60992> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK: [[CMX_1_MIN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <60992> -> memref<1x1x1x1xf32, [@CMX_NN, 1]>
    // CHECK: [[MIN_DIST:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <60992> -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[MAX:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1x1x1xf32, @DDR>
    // CHECK: [[CMX_0_MAX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <128> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK: [[CMX_1_MAX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <128> -> memref<1x1x1x1xf32, [@CMX_NN, 1]>
    // CHECK: [[MAX_DIST:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <128> -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK: [[DATA_IN_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <7468800> -> memref<1x1x333x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK: [[DATA_IN_1:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <8001600> -> memref<1x1x333x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK: [[CMX_0_DATA:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <194624> -> memref<1x1x333x400xf32, [@CMX_NN, 0]>
    // CHECK: [[CMX_1_DATA:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <194624> -> memref<1x1x333x400xf32, [@CMX_NN, 1]>
    // CHECK: [[DATA_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <194624> -> memref<1x1x333x400xf32, [@CMX_NN, 0]>
    // CHECK: [[DATA_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <194624> -> memref<1x1x333x400xf32, [@CMX_NN, 1]>
    // CHECK: [[CMX_0_DATA_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <995904> -> memref<1x1x333x400xui8, [@CMX_NN, 0]>
    // CHECK: [[CMX_1_DATA_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <995904> -> memref<1x1x333x400xui8, [@CMX_NN, 1]>
    // CHECK: [[OUT_CMX_0_DATA:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <995904> -> memref<1x1x333x400xui8, [@CMX_NN, 0]>
    // CHECK: [[OUT_CMX_1_DATA:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <995904> -> memref<1x1x333x400xui8, [@CMX_NN, 1]>
    // CHECK: [[CMX_0_SCALE_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <727424> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK: [[OUT_CMX_0_SCALE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <727424> -> memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK: [[OUT_CMX_1_SCALE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <727424> -> memref<1x1x1x1xf32, [@CMX_NN, 1]>
    // CHECK: [[CMX_0_ZP_BUFF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <727488> -> memref<1x1x1x1xui8, [@CMX_NN, 0]>
    // CHECK: [[OUT_CMX_0_ZP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <727488> -> memref<1x1x1x1xui8, [@CMX_NN, 0]>
    // CHECK: [[OUT_CMX_1_ZP:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <727488> -> memref<1x1x1x1xui8, [@CMX_NN, 1]>
    // CHECK: [[DDR_DATA:%.+]]  = VPURT.DeclareBuffer <NetworkOutput> [0] <1334000> -> memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK: [[DDR_DATA_0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <1334000> -> memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK: [[DDR_DATA_1:%.+]]  = VPURT.DeclareBuffer <NetworkOutput> [0] <1467200> -> memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK: [[DDR_SCALE:%.+]]  = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<1x1x1x1xf32, @DDR>
    // CHECK: [[DDR_ZP:%.+]]  = VPURT.DeclareBuffer <NetworkOutput> [2] <0> -> memref<1x1x1x1xui8, @DDR>

    // CHECK: VPUIP.NNDMA <{port = 0 : i64}> inputs([[MIN]] : memref<1x1x1x1xf32, [@CMX_NN, 0]>) outputs([[MIN_DIST]] : !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: VPUIP.NNDMA <{port = 1 : i64}> inputs([[MAX]] : memref<1x1x1x1xf32, @DDR>) outputs([[MAX_DIST]] : !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<1x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK: VPUIP.NNDMA <{port = 0 : i64}> inputs([[DATA_IN_0]] : memref<1x1x333x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) outputs([[DATA_CMX_0]] : memref<1x1x333x400xf32, [@CMX_NN, 0]>) -> memref<1x1x333x400xf32, [@CMX_NN, 0]>

    // CHECK: VPUIP.NNDMA <{port = 1 : i64}> inputs([[DATA_IN_1]] : memref<1x1x333x400xf32, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) outputs([[DATA_CMX_1]] : memref<1x1x333x400xf32, [@CMX_NN, 1]>) -> memref<1x1x333x400xf32, [@CMX_NN, 1]>

    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>}
    // CHECK-SAME:  @VPU.SW::@builtin_DynamicQuantize
    // CHECK-SAME:  inputs([[CMX_0_DATA]] as [[ARG_0:[^:]+]]: memref<1x1x333x400xf32, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[CMX_0_MIN]] as [[ARG_1:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 0]>,
    // CHECK-SAME:         [[CMX_0_MAX]] as [[ARG_2:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[OUT_CMX_0_DATA]] as [[ARG_3:[^:]+]]: memref<1x1x333x400xui8, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[OUT_CMX_0_SCALE]] as [[ARG_4:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 0]>,
    // CHECK-SAME:               [[OUT_CMX_0_ZP]] as [[ARG_5:[^:]+]]: memref<1x1x1x1xui8, [@CMX_NN, 0]>) on tile 0
    // CHECK:       -> (memref<1x1x333x400xui8, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x1x1xui8, [@CMX_NN, 0]>)
    // CHECK:       VPUIP.SW.Kernel.run([[ARG_0]], [[ARG_1]], [[ARG_2]], [[ARG_3]], [[ARG_4]], [[ARG_5]])
    // CHECK:       memref<1x1x333x400xf32, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:       memref<1x1x333x400xui8, [@CMX_NN, 0]>, memref<1x1x1x1xf32, [@CMX_NN, 0]>, memref<1x1x1x1xui8, [@CMX_NN, 0]>

    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 3, 0, 0>}
    // CHECK-SAME:  @VPU.SW::@builtin_DynamicQuantize
    // CHECK-SAME:  inputs([[CMX_1_DATA]] as [[ARG_0:[^:]+]]: memref<1x1x333x400xf32, [@CMX_NN, 1]>,
    // CHECK-SAME:         [[CMX_1_MIN]] as [[ARG_1:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 1]>,
    // CHECK-SAME:         [[CMX_1_MAX]] as [[ARG_2:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 1]>)
    // CHECK-SAME:       outputs([[OUT_CMX_1_DATA]] as [[ARG_3:[^:]+]]: memref<1x1x333x400xui8, [@CMX_NN, 1]>,
    // CHECK-SAME:               [[OUT_CMX_1_SCALE]] as [[ARG_4:[^:]+]]: memref<1x1x1x1xf32, [@CMX_NN, 1]>,
    // CHECK-SAME:               [[OUT_CMX_1_ZP]] as [[ARG_5:[^:]+]]: memref<1x1x1x1xui8, [@CMX_NN, 1]>) on tile 1
    // CHECK:       -> (memref<1x1x333x400xui8, [@CMX_NN, 1]>, memref<1x1x1x1xf32, [@CMX_NN, 1]>, memref<1x1x1x1xui8, [@CMX_NN, 1]>)
    // CHECK:       VPUIP.SW.Kernel.run([[ARG_0]], [[ARG_1]], [[ARG_2]], [[ARG_3]], [[ARG_4]], [[ARG_5]])
    // CHECK:       memref<1x1x333x400xf32, [@CMX_NN, 1]>, memref<1x1x1x1xf32, [@CMX_NN, 1]>, memref<1x1x1x1xf32, [@CMX_NN, 1]>
    // CHECK:       memref<1x1x333x400xui8, [@CMX_NN, 1]>, memref<1x1x1x1xf32, [@CMX_NN, 1]>, memref<1x1x1x1xui8, [@CMX_NN, 1]>

    // CHECK: VPUIP.NNDMA <{port = 0 : i64}> inputs([[CMX_0_DATA_BUFF]] : memref<1x1x333x400xui8, [@CMX_NN, 0]>) outputs([[DDR_DATA_0]] : memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) -> memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK:  VPUIP.NNDMA <{port = 1 : i64}> inputs([[CMX_1_DATA_BUFF]] : memref<1x1x333x400xui8, [@CMX_NN, 1]>) outputs([[DDR_DATA_1]] : memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>) -> memref<1x1x333x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>
    // CHECK:  VPUIP.NNDMA <{port = 0 : i64}> inputs([[CMX_0_SCALE_BUFF]] : memref<1x1x1x1xf32, [@CMX_NN, 0]>) outputs([[DDR_SCALE]] : memref<1x1x1x1xf32, @DDR>) -> memref<1x1x1x1xf32, @DDR>
    // CHECK:  VPUIP.NNDMA <{port = 1 : i64}> inputs([[CMX_0_ZP_BUFF]] : memref<1x1x1x1xui8, [@CMX_NN, 0]>) outputs([[DDR_ZP]] : memref<1x1x1x1xui8, @DDR>) -> memref<1x1x1x1xui8, @DDR>
    // CHECK:  return [[DDR_DATA]], [[DDR_SCALE]], [[DDR_ZP]]
    // CHECK:  memref<1x1x666x400xui8, {order = #NCHW, strides = [2400000, 2400000, 400, 1]}, @DDR>, memref<1x1x1x1xf32, @DDR>, memref<1x1x1x1xui8, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x171x512xf16,    {
        order = #NCHW, strides = [1400832, 87552, 512, 1]
    }, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!Input_CMX = memref<1x3x171x512xf16, {order = #NCHW, strides = [1400832, 87552, 512, 1]}, @CMX_NN>
!Output_DDR = memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>

//CHECK-LABEL: @UnrollNNDMACMX2DDR_NCHW
func.func @UnrollNNDMACMX2DDR_NCHW() -> !Output_DDR {

    %input = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer <DDR> <2010112> -> memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %162 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%input : !InputDistributed) outputs(%output : !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR
  }


//CHECK: [[BUF_0:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x86x512xf16, {order = #NCHW, strides = [704512, 44032, 512, 1]}, [@CMX_NN, 0]>
//CHECK: [[BUF_1:%.+]]   = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x85x512xf16, {order = #NCHW, strides = [696320, 43520, 512, 1]}, [@CMX_NN, 1]>
//CHECK: [[OUTBUF:%.+]]  = VPURT.DeclareBuffer <DDR> <2010112> -> memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[OUTBUF1:%.+]] = VPURT.DeclareBuffer <DDR> <2010112> -> memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[OUTBUF2:%.+]] = VPURT.DeclareBuffer <DDR> <2098176> -> memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[BAR_0:%.+]]   = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
//CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
//CHECK: VPUIP.NNDMA <{port = 0 : i64}> inputs([[BUF_0]] : memref<1x3x86x512xf16, {order = #NCHW, strides = [704512, 44032, 512, 1]}, [@CMX_NN, 0]>) outputs([[OUTBUF1]] : memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>) -> memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) {
//CHECK: VPUIP.NNDMA <{port = 1 : i64}> inputs([[BUF_1]] : memref<1x3x85x512xf16, {order = #NCHW, strides = [696320, 43520, 512, 1]}, [@CMX_NN, 1]>) outputs([[OUTBUF2]] : memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>) -> memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: return [[OUTBUF]] : memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPUIP.DistributedBuffer<1x16x112x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
!Output_CMX = !VPUIP.DistributedBuffer<1x16x112x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    , @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<1x16x112x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x112x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
// CHECK-LABEL: @AdjustSizeAlignment
func.func @AdjustSizeAlignment(%output: !Output_CMX) -> !Output_CMX {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !Input_CMX
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !Output_CMX

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !Input_CMX) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !Output_CMX) -> !Output_CMX
    }
    return %18 : !Output_CMX

    // Check alignment is set back to 512 for per cluster buffers
    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPUIP.DistributedBuffer<1x16x111x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
!Output_CMX = !VPUIP.DistributedBuffer<1x16x111x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    , @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<1x16x111x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x111x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
// CHECK-LABEL: @AdjustSizeAlignmentUnequalPerClusterShapes
func.func @AdjustSizeAlignmentUnequalPerClusterShapes(%output: !Output_CMX) -> !Output_CMX {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !Input_CMX
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !Output_CMX

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !Input_CMX) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !Output_CMX) -> !Output_CMX
    }
    return %18 : !Output_CMX

    // Check alignment is set back to 512 for per cluster buffers
    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x55x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x56x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x55x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPUIP.DistributedBuffer<1x16x112x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!Output_CMX = !VPUIP.DistributedBuffer<1x16x112x112xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    , @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

!Compact_DDR = memref<1x16x112x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<1x16x112x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// When spilled buffer in DDR needed more size than total size of swizzling aligned per cluster buffers we adjust the alignment
// to meet memory demand in DDR, during unrolling this should be readjusted so the per cluster buffer don't carry parent's alignment
// CHECK-LABEL: @AdjustSizeAlignmentHigherClusters
func.func @AdjustSizeAlignmentHigherClusters(%output: !Output_CMX) -> !Output_CMX {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <851968> {swizzlingKey = 5 : i64} -> !Input_CMX
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <1409024> {swizzlingKey = 5 : i64} -> !Output_CMX

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !Input_CMX) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !Output_CMX) -> !Output_CMX
    }
    return %18 : !Output_CMX

    // Check alignment is set back to 512 for per cluster buffers
    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <50176> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_2:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_3:%.+]] = VPURT.DeclareBuffer <DDR> <150528> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <50176> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_2:%.+]] = VPURT.DeclareBuffer <DDR> <100352> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_3:%.+]] = VPURT.DeclareBuffer <DDR> <150528> {swizzlingKey = 5 : i64} -> memref<1x16x28x112xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input_CMX = !VPUIP.DistributedBuffer<256x384x3x3xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>},
    @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
!Output_CMX = !VPUIP.DistributedBuffer<256x384x3x3xi8,
    {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}
    , @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>

!Compact_DDR = memref<256x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @DDR>
!Compact_CMX = memref<256x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 1024 : i64>}, @CMX_NN>

// CHECK-LABEL: @AdjustSizeAlignmentHigherBatches
func.func @AdjustSizeAlignmentHigherBatches(%output: !Output_CMX) -> !Output_CMX {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %13 = VPURT.DeclareBuffer <CMX_NN> <0> {swizzlingKey = 5 : i64} -> !Input_CMX
    %17 = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> !Compact_DDR
    %18 = VPURT.DeclareBuffer <CMX_NN> <884736> {swizzlingKey = 5 : i64} -> !Output_CMX

    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%13 : !Input_CMX) outputs(%17 : !Compact_DDR) -> !Compact_DDR
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
        %47 = VPUIP.NNDMA <{port = 0 : i64, spillId = 0 : i64}> inputs(%17 : !Compact_DDR) outputs(%18 : !Output_CMX) -> !Output_CMX
    }
    return %18 : !Output_CMX

    //CHECK: [[DDR0_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<128x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR0_1:%.+]] = VPURT.DeclareBuffer <DDR> <442368> {swizzlingKey = 5 : i64} -> memref<128x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> {swizzlingKey = 5 : i64} -> memref<128x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
    //CHECK: [[DDR1_1:%.+]] = VPURT.DeclareBuffer <DDR> <442368> {swizzlingKey = 5 : i64} -> memref<128x384x3x3xi8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, @DDR>
}
