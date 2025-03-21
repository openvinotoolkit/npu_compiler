//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --unwrap-cluster-tiling  %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#C = affine_map<(d0) -> (d0)>
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

!Input_DDR = memref<1x16x33x32xf16, #NHWC, @DDR>
!Output_DDR = memref<1x16x33x32xf16, #NHWC, @DDR>
!Weights_DDR = memref<16x16x1x1xf16, #NHWC>
!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

!Profiling_CMX = memref<4xui64, [@CMX_NN, 0]>
!Profiling_DDR = memref<4xui64>

VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
module @VPU.SW {
    func.func private @builtin_TanhOp(memref<*xf16>, memref<*xf16>, i64) attributes {VPU.kernel_code = "activation_tanh.cpp", VPU.kernel_entry = "activation_tanh"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

//CHECK-LABEL: @UnwrapDmaActShaveNCEWithProfiling
func.func @UnwrapDmaActShaveNCEWithProfiling(%input: !Input_DDR, %output: !Output_DDR, %prof_output: !Profiling_DDR) -> (!Output_DDR, !Profiling_DDR) {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, #NHWC> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR
    %profiling_out = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> !Profiling_DDR

    // CMX buffers
    %nce_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %nce_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !WeightsTableDistributed

    %act_in_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %act_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed

    // Profiling buffers
    %prof_buffer_cmx = VPURT.DeclareBuffer <CMX_NN> <353584> -> !ProfilingDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%parent_in as %arg0: !Input_DDR)
                outputs(%nce_input_cmx as %arg1: !InputStub_CMX) -> !InputDistributed {
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

    // Cluster tiling with NCEClusterTask
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
         %0:2 = VPUIP.NCEClusterTiling
                 inputs(%nce_input_cmx as %arg0: !InputStub_CMX,
                         %weights as %arg1: !WeightsStub_CMX,
                         %weights_table as %arg2: !WeightsTableStub_CMX)
                 outputs(%nce_out_cmx as %arg3: !OutputStub_CMX,
                         %prof_buffer_cmx as %arg4: !Profiling_CMX)
                     -> (!OutputDistributed, !ProfilingDistributed)  {

               %1:2 = VPUIP.NCEClusterTask {
                         kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = #VPUIP.nce_task_type<CONV>
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
    }

    // Cluster tiling with ActShave
    VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) {
      %0 = VPUIP.NCEClusterTiling inputs(%act_in_cmx as %arg0: !OutputStub_CMX) outputs(%act_out_cmx as %arg1: !OutputStub_CMX) -> !OutputDistributed {
         %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>}
         @VPU.SW::@builtin_TanhOp inputs(%arg0 as %arg2: !OutputStub_CMX) outputs(%arg1 as %arg3: !OutputStub_CMX) on tile 0 -> !OutputStub_CMX  {
            VPUIP.SW.Kernel.run {attrs = [0]}(%arg2, %arg3) : !OutputStub_CMX, !OutputStub_CMX
         }
      }
    }

    // Copyback output
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
         %0 = VPUIP.NCEClusterTiling inputs(%act_out_cmx as %arg0: !OutputStub_CMX)
                outputs(%parent_out as %arg1: !Output_DDR) -> !Output_DDR {
             VPUIP.NNDMA inputs(%arg0: !OutputStub_CMX) outputs(%arg1: !Output_DDR) -> !Output_DDR
         }
    }

    // Copyback profiling
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
        %0 = VPUIP.NCEClusterTiling inputs(%prof_buffer_cmx as %arg0: !Profiling_CMX)
                outputs(%profiling_out as %arg1: !Profiling_DDR) -> !Profiling_DDR {
             VPUIP.NNDMA inputs(%arg0: !Profiling_CMX) outputs(%arg1: !Profiling_DDR) -> !Profiling_DDR
         }
    }

    return %output, %prof_output: !Output_DDR, !Profiling_DDR

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR2:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[WEIGHTS_CST:%.*]] = const.Declare memref<16x16x1x1xf16, #NHWC>
    //CHECK:    [[WEIGHTS_TABLE_CST:%.*]] = const.Declare memref<16x1x1x4xsi32>

    //CHECK:    [[NET_IN:%.*]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x33x32xf16, #NHWC, @DDR>
    //CHECK:    [[NET_OUT:%.*]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x33x32xf16, #NHWC, @DDR>
    //CHECK:    [[PROF_OUT:%.*]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui64>
    //CHECK:    [[NCE_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[NCE_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <34816> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[WEIGHTS_TABLE_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[ACT_SHAVE_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[ACT_SHAVE_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> <17408> -> !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    //CHECK:    [[PROF_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> <353584> -> !VPUIP.DistributedBuffer<4xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[NET_IN]] : memref<1x16x33x32xf16, #NHWC, @DDR>)
    //CHECK-SAME:       outputs([[NCE_IN]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, #NHWC>)
    //CHECK-SAME:       outputs([[WEIGHTS_BUF]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE_BUF]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // NCEClusterTask
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NCEClusterTask
    //CHECK-SAME:       input([[NCE_IN]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           weights([[WEIGHTS_BUF]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE_BUF]] : !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_input([[NCE_IN]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[NCE_OUT]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[NCE_OUT]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           profiling_data([[PROF_BUF]] : !VPUIP.DistributedBuffer<4xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0
    //CHECK:                DPUTask {cluster_id = 1
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // ActShaveTask
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.SW.Kernel
    //CHECK-SAME:       inputs([[ACT_SHAVE_IN]]
    //CHECK-SAME:       outputs([[ACT_SHAVE_OUT]]
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[ACT_SHAVE_OUT]] : !VPUIP.DistributedBuffer<1x16x33x32xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       outputs([[NET_OUT]] : memref<1x16x33x32xf16, #NHWC, @DDR>)
    //CHECK:        }

    // Copyback profiling
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) {
    //CHECK-NOT:      VPUIP.NCEClusterTiling
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[PROF_BUF]] : !VPUIP.DistributedBuffer<4xui64, #C, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2], num_clusters = 2 : i64}>)
    //CHECK-SAME:       outputs([[PROF_OUT]] : memref<4xui64>)
    //CHECK:        }
}
