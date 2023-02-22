//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --unroll-cluster-tiling  %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC, @DDR>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>

!Buffer0_CMX = type memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
!Buffer1_CMX = type memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

//CHECK-LABEL: @UnrollNNDMA
func @UnrollNNDMA(%input: memref<1x16x33x32xf16>, %output: memref<1x16x33x32xf16>) -> memref<1x16x33x32xf16> {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "DDR" <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "DDR" <33792> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %input1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> !Buffer0_CMX
    %input2 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> !Buffer1_CMX

    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %output1 = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> !Buffer0_CMX
    %output2 = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> !Buffer1_CMX

    %weights = VPURT.DeclareBuffer "CMX_NN" <34816> -> !WeightsDistributed

    // Reorder input

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NHWC}
            inputs(%input: memref<1x16x33x32xf16>)
            outputs(%parent_in: !Input_DDR)
            -> !Input_DDR
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload input
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Simulate 1st task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%input1: !Buffer0_CMX) outputs(%output1: !Buffer0_CMX) -> !Buffer0_CMX
    }

    // Simulate 2nd task
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%input2: !Buffer1_CMX) outputs(%output2: !Buffer1_CMX) -> !Buffer1_CMX
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    // Reorder output

    VPURT.Task waits(%bar3: !VPURT.Barrier) {
        VPUIP.PermuteUPA {order_value = #NCHW}
            inputs(%parent_out: !Output_DDR)
            outputs(%output: memref<1x16x33x32xf16>)
            -> memref<1x16x33x32xf16>
    }

    return %output: memref<1x16x33x32xf16>

    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR3:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <33792> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "DDR" <51200> -> memref<1x16x16x32xf16, #NHWC, @DDR>

    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Simulate tasks
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier)  {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Copyback 1st part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Copyback 2nd part of output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK:        }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>
!Output_CMX = type memref<1x16x33x32xf16, #NHWC, [@CMX_NN, 0]>
!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>

//CHECK-LABEL: @UnrollNNDMACMX2CMX
func @UnrollNNDMACMX2CMX() -> !Output_CMX {
    %input = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer "CMX_NN" [0] <33792> -> !Output_CMX
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !InputStub_CMX)
                outputs(%output as %arg1: !Output_CMX) -> !Output_CMX{
             VPUIP.NNDMA inputs(%arg0: !InputStub_CMX) outputs(%arg1: !Output_CMX) -> !Output_CMX
         }
    }

    return %output: !Output_CMX


    //CHECK: [[BUF_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: [[BUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK: [[OUTBUF:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33792> -> memref<1x16x33x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: [[OUTBUF_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <33792> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: [[OUTBUF_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <51200> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: [[BAR_0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:    VPUIP.NNDMA {port = 0 : i64} inputs([[BUF_0]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>) outputs([[OUTBUF_0]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: }
    //CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    //CHECK:    VPUIP.NNDMA {port = 1 : i64} inputs([[BUF_1]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>) outputs([[OUTBUF_1]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK: }
    //CHECK: return [[OUTBUF]] : memref<1x16x33x32xf16, #NHWC, [@CMX_NN, 0]>
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ProfilingDistributed = type !VPUIP.DistributedBuffer<
    4xui64, affine_map<(d0) -> (d0)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2],
    num_clusters = 2 : i64
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

!Profiling_CMX = type memref<4xui64, [@CMX_NN, 0]>
!Profiling_DDR = type memref<4xui64>

//CHECK-LABEL: @UnrollNCEWithProfiling
func @UnrollNCEWithProfiling(%input: !Input_DDR, %output: !Output_DDR, %prof_output: !Profiling_DDR) -> (!Output_DDR, !Profiling_DDR) {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR
    %profiling_out = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> !Profiling_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !WeightsTableDistributed

    // Profiling buffers
    %prof_buffer_cmx = VPURT.DeclareBuffer "CMX_NN" <353584> -> !ProfilingDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0:2 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX,
                         %prof_buffer_cmx as %arg4: !Profiling_CMX)
                     -> (!OutputStub_CMX, !Profiling_CMX)  {

               %1:2 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                         profiling_data(%arg4: !Profiling_CMX)
                             -> !OutputStub_CMX, !Profiling_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 17, 0], outEnd = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    // Copyback profiling
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%prof_buffer_cmx as %arg0: !Profiling_CMX)
                outputs(%profiling_out as %arg1: !Profiling_DDR) -> !Profiling_DDR {
             VPUIP.NNDMA inputs(%arg0: !Profiling_CMX) outputs(%arg1: !Profiling_DDR) -> !Profiling_DDR
         }
    }

    return %output, %prof_output: !Output_DDR, !Profiling_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:    [[PROF1_DDR:%.*]] = VPURT.DeclareBuffer "ProfilingOutput" [0] <0> -> memref<2xui64, @DDR>
    //CHECK:    [[PROF2_DDR:%.*]]  = VPURT.DeclareBuffer "ProfilingOutput" [0] <16> -> memref<2xui64, @DDR>


    //CHECK:    [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[PROF1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <353584> -> memref<2xui64, [@CMX_NN, 0]>
    //CHECK:    [[PROF2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <353584> -> memref<2xui64, [@CMX_NN, 1]>
    //CHECK:    [[PROF1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <353584> -> memref<2xui64, [@CMX_NN, 0]>
    //CHECK:    [[PROF2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <353584> -> memref<2xui64, [@CMX_NN, 1]>


    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           profiling_data([[PROF1_CMX]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           profiling_data([[PROF2_CMX]] : memref<2xui64, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[PROF1_CMX_COPY]] : memref<2xui64, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[PROF1_DDR]] : memref<2xui64, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[PROF2_CMX_COPY]] : memref<2xui64, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[PROF2_DDR]] : memref<2xui64, @DDR>)
    //CHECK:        }

    //CHECK:    return %arg1, %arg2 : memref<1x16x33x32xf16, #NHWC, @DDR>, memref<4xui64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x3x3xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

//CHECK-LABEL: @UnrollNCESegmentedConv
func @UnrollNCESegmentedConv(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x3x3xf16, #NHWC> =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx    = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <39424> -> !WeightsTableDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                         kernel_size = [3, 3],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 17, 0], outEnd = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x3x3xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK-DAG:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>

    //CHECK-DAG:    [[PARENT_IN_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK-DAG:    [[PARENT_OUT_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK-DAG:    [[WEIGHTS1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <39424> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <39424> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>


    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }


    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x33x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!WeightsDistributed = type !VPUIP.DistributedBuffer<
    16x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!WeightsTableDistributed = type !VPUIP.DistributedBuffer<
    16x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!Input_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = type memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = type memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = type memref<16x1x1x4xsi32>

!InputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = type memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = type memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = type memref<16x1x1x4xsi32, @CMX_NN>

//CHECK-LABEL: @UnrollNCESequence
func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer "CMX_NN" <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !WeightsTableDistributed

    %parent_input_cmx_2 = VPURT.DeclareBuffer "CMX_NN" <17408> -> !InputDistributed
    %parent_out_cmx_2 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%parent_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
             VPUIP.NNDMA inputs(%arg0: !Input_DDR) outputs(%arg1: !InputStub_CMX) -> !InputStub_CMX
         }
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_cst as %arg0: !Weights_DDR)
                outputs(%weights as %arg1: !WeightsStub_CMX) -> !WeightsDistributed {
             VPUIP.NNDMA inputs(%arg0: !Weights_DDR) outputs(%arg1: !WeightsStub_CMX) -> !WeightsStub_CMX
         }
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%weights_table_cst as %arg0: !WeightsTable_DDR)
                outputs(%weights_table as %arg1: !WeightsTableStub_CMX) -> !WeightsTableDistributed {
             VPUIP.NNDMA inputs(%arg0: !WeightsTable_DDR) outputs(%arg1: !WeightsTableStub_CMX) -> !WeightsTableStub_CMX
         }
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 17, 0], outEnd = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Cluster tiling
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling
                 inputs(%parent_input_cmx_2 as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%parent_out_cmx_2 as %arg3: !OutputStub_CMX)
                     -> !OutputStub_CMX {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = "CONV"
                     }  input(%arg0 : !InputStub_CMX)
                         weights(%arg1 : !WeightsStub_CMX)
                         weight_table(%arg2 : !WeightsTableStub_CMX)
                         parent_input(%arg0 : !InputStub_CMX)
                         parent_output(%arg3 : !OutputStub_CMX)
                         outputs(%arg3 : !OutputStub_CMX)
                             -> !OutputStub_CMX variants :  {
                            DPUTask {
                                outStart = [0, 0, 0], outEnd = [31, 16, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 0 : i64
                            }
                            DPUTask {
                                outStart = [0, 17, 0], outEnd = [31, 32, 31],
                                mpe_mode = "VECTOR_FP16",
                                pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
                                cluster_id = 1 : i64
                            }
                         } PPE :  {
                         }
        }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_out_cmx_2 as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    return %output: !Output_DDR

    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>
    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[IN2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT1_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x16x17x32xf16, #NHWC, @DDR>
    //CHECK:    [[OUT2_DDR:%.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <17408> -> memref<1x16x16x32xf16, #NHWC, @DDR>

    //CHECK:    [[PARENT_IN_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[PARENT_OUT_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[OUT1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    //CHECK:    [[WEIGHTS1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE1_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS_TABLE2_CMX_1ST_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <35328> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS_TABLE_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    [[PARENT_IN_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[IN1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <17408> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <17408> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[PARENT_OUT_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>
    //CHECK:    [[OUT1_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_2ND_TASK:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_CMX_COPY]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // 1st task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_1ST_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 1st task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_1ST_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_1ST_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_1ST_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_1ST_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }


    // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX_2ND_TASK]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 16, 31], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:           {is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], task_type = "CONV"}
    //CHECK-SAME:       input([[IN2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX_2ND_TASK]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX_2ND_TASK]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX_2ND_TASK]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = "VECTOR_FP16", outEnd = [31, 32, 31], outStart = [0, 17, 0],
    //CHECK-SAME:               pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 0 : i64}
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x17x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
    //CHECK:          VPUIP.NNDMA {port = 1 : i64}
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x16x32xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x16x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    //CHECK:    return %arg1 : memref<1x16x33x32xf16, #NHWC, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!typeCmxDistributed = type !VPUIP.DistributedBuffer<
    1x4x512x1xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2
}>

!type_CMX_memref = type memref<1x4x512x1xf16, #NCHW, @CMX_NN>


!Input_DDR  = type memref<1x4x512x1xf16, #NCHW, @DDR>
!Output_DDR = type memref<1x4x512x1xf16, #NCHW, @DDR>


VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "tanh_fp16.cpp", VPU.kernel_entry = "tanh_fp16"}
    func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @UnrollTanhSOH
func @UnrollTanhSOH(%input0: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %395 = VPURT.DeclareBuffer "DDR" <4096> -> !Input_DDR
    %302 = VPURT.DeclareBuffer "DDR" <0> -> !Output_DDR

    %300 = VPURT.DeclareBuffer "CMX_NN" <0> -> !typeCmxDistributed
    %301 = VPURT.DeclareBuffer "CMX_NN" <2048> -> !typeCmxDistributed


    VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {cycleBegin = 1907728 : i64, cycleEnd = 1907899 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%395 as %arg2: !Input_DDR) outputs(%300 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !Input_DDR) outputs(%arg3 : !type_CMX_memref) -> !type_CMX_memref
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 1907899 : i64, cycleEnd = 1907901 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%300 as %arg2: !type_CMX_memref) outputs(%301 as %arg3: !type_CMX_memref) -> !typeCmxDistributed {
         %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>}
         @VPU.SW::@builtin_TanhOp inputs(%arg2 as %arg4: !type_CMX_memref) outputs(%arg3 as %arg5: !type_CMX_memref) on tile 0 -> !type_CMX_memref  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg4, %arg5) : !type_CMX_memref, !type_CMX_memref
         }
      }
    }
    VPURT.Task waits(%bar1 : !VPURT.Barrier)  attributes {cycleBegin = 1907901 : i64, cycleEnd = 1908072 : i64, isTrailingSWLayer = false} {
      %398 = VPUIP.NCEClusterTiling inputs(%301 as %arg2 : !type_CMX_memref) outputs(%302 as %arg3: !Output_DDR) -> !Output_DDR {
        %399 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : !type_CMX_memref) outputs(%arg3 : !Output_DDR) -> !Output_DDR
      }
    }


    return %output: !Output_DDR

    //CHECK:    [[IN_DDR1:%.*]] = VPURT.DeclareBuffer "DDR" <4096> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[IN_DDR2:%.*]] = VPURT.DeclareBuffer "DDR" <4608> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[OUT_DDR1:%.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[OUT_DDR2:%.*]] = VPURT.DeclareBuffer "DDR" <512> -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    [[IN0_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[IN0_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[IN1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUT0_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUT0_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUT1_CMX_COPY:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <2048> -> memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.NNDMA {port = 0 : i64} inputs([[IN_DDR1]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs([[IN0_CMX_COPY]] : memref<1x4x256x1xf16, [@CMX_NN, 0]>) -> memref<1x4x256x1xf16, [@CMX_NN, 0]>
    //CHECK:    VPUIP.NNDMA {port = 1 : i64} inputs([[IN_DDR2]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) outputs([[IN1_CMX_COPY]] : memref<1x4x256x1xf16, [@CMX_NN, 1]>) -> memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_TanhOp inputs([[IN0_CMX]] as [[ARG2:%.*]]: memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs([[OUT0_CMX_COPY]] as [[ARG3:%.*]]: memref<1x4x256x1xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x4x256x1xf16, [@CMX_NN, 0]>{
    //CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[ARG2]], [[ARG3]]) : memref<1x4x256x1xf16, [@CMX_NN, 0]>, memref<1x4x256x1xf16, [@CMX_NN, 0]>

    //CHECK:    %results = VPUIP.SW.Kernel {result_segment_sizes = dense<[1, 0]> : vector<2xi32>} @VPU.SW::@builtin_TanhOp inputs([[IN1_CMX]] as [[ARG22:%.*]]: memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs([[OUT1_CMX_COPY]] as [[ARG23:%.*]]: memref<1x4x256x1xf16, [@CMX_NN, 1]>) on tile 1 -> memref<1x4x256x1xf16, [@CMX_NN, 1]>{
    //CHECK:    VPUIP.SW.Kernel.run {attrs = [0]}([[ARG22]], [[ARG23]]) : memref<1x4x256x1xf16, [@CMX_NN, 1]>, memref<1x4x256x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPUIP.NNDMA {port = 0 : i64} inputs([[OUT0_CMX]] : memref<1x4x256x1xf16, [@CMX_NN, 0]>) outputs([[OUT_DDR1]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
    //CHECK:    VPUIP.NNDMA {port = 1 : i64} inputs([[OUT1_CMX]] : memref<1x4x256x1xf16, [@CMX_NN, 1]>) outputs([[OUT_DDR2]] : memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>) -> memref<1x4x256x1xf16, {order = #NCHW, strides = [2048, 512, 1, 1]}, @DDR>
}


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = type !VPUIP.DistributedBuffer<
    1x3x171x512xf16,    {
        order = #NCHW, strides = [1400832, 87552, 512, 1]
    }, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [1, 1, 2, 1],
        num_clusters = 2 : i64
}>

!Input_CMX =  type memref<1x3x171x512xf16, {order = #NCHW, strides = [1400832, 87552, 512, 1]}, @CMX_NN>
!Output_DDR = type memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>

//CHECK-LABEL: @UnrollNNDMACMX2DDR_NCHW
func @UnrollNNDMACMX2DDR_NCHW() -> !Output_DDR {

    %input = VPURT.DeclareBuffer "CMX_NN" <0> -> !InputDistributed
    %output = VPURT.DeclareBuffer "DDR" <2010112> -> memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %161 = VPUIP.NCEClusterTiling inputs(%input as %arg0: !Input_CMX) outputs(%output as %arg1: !Output_DDR) -> !Output_DDR {
        %162 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : !Input_CMX) outputs(%arg1 : !Output_DDR) -> !Output_DDR
      }
    }

    return %output: !Output_DDR
  }


//CHECK: [[BUF_0:%.*]]   = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x3x86x512xf16, [@CMX_NN, 0]>
//CHECK: [[BUF_1:%.*]]   = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<1x3x85x512xf16, [@CMX_NN, 1]>
//CHECK: [[OUTBUF:%.*]]  = VPURT.DeclareBuffer "DDR" <2010112> -> memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[OUTBUF1:%.*]] = VPURT.DeclareBuffer "DDR" <2010112> -> memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[OUTBUF2:%.*]] = VPURT.DeclareBuffer "DDR" <2098176> -> memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: [[BAR_0:%.*]]   = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
//CHECK: VPURT.Task updates([[BAR_0]] : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
//CHECK: VPUIP.NNDMA {port = 0 : i64} inputs([[BUF_0]] : memref<1x3x86x512xf16, [@CMX_NN, 0]>) outputs([[OUTBUF1]] : memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>) -> memref<1x3x86x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
//CHECK: VPUIP.NNDMA {port = 1 : i64} inputs([[BUF_1]] : memref<1x3x85x512xf16, [@CMX_NN, 1]>) outputs([[OUTBUF2]] : memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>) -> memref<1x3x85x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
//CHECK: return [[OUTBUF]] : memref<1x3x171x512xf16, {order = #NCHW, strides = [786432, 262144, 512, 1]}, @DDR>
