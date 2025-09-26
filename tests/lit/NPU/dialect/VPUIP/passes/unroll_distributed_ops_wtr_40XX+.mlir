//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file  --init-compiler="vpu-arch=%arch%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x128x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x64x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 32, 4, 1], [1, 32, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 4, 1], [1, 64, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!Input_DDR = memref<1x128x4x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x64x4x1xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x128x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, @DDR>

//CHECK-LABEL: @OptimizeWeightTableDMAsNCETask
func.func @OptimizeWeightTableDMAsNCETask(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // Constants
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // CMX Buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <1171200> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <1150464> -> !ParentOutputDistributed
    %wt_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !WeightsTableDistributed
    %weights_cmx = VPURT.DeclareBuffer <CMX_NN> <3584> -> !WeightsDistributed

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weight table
    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table : !WeightsTable_DDR) outputs(%wt_cmx : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights : !Weights_DDR) outputs(%weights_cmx : !WeightsDistributed) -> !WeightsDistributed
    }

    // Cluster task
    VPURT.Task waits(%bar2, %bar1, %bar0 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
        %49 = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6233 : i64,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>
                }
        input(%parent_input_cmx : !ParentInputDistributed)
        weights(%weights_cmx : !WeightsDistributed)
        weight_table(%wt_cmx : !WeightsTableDistributed)
        parent_input(%parent_input_cmx : !ParentInputDistributed)
        parent_output(%parent_out_cmx : !ParentOutputDistributed)
        outputs(%parent_out_cmx : !ParentOutputDistributed) -> !ParentOutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 31], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 63], outStart = [0, 0, 32],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WT:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [32, 1, 1, 4]>]

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[WT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0>
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {
    //CHECK-SAME:               mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    //CHECK-SAME:               alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:      compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:      memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

    //CHECK:    VPURT.Task updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64} inputs([[WT]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_CMX]] : !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
    //CHECK-SAME:           -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN,
    //CHECK-SAME:               {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    //CHECK-SAME{LITERAL}:               compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    //CHECK-SAME{LITERAL}:               memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_2]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weight_table([[WT_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }

    //CHECK:     VPURT.Task waits([[BAR_2]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 32 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weight_table([[WT_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x128x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x64x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 32, 4, 1], [1, 32, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 4, 1], [1, 64, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!Input_DDR = memref<1x128x4x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x64x4x1xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x128x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, @DDR>

//CHECK-LABEL: @DoNotOptimizeDMAsForWTSpill
func.func @DoNotOptimizeDMAsForWTSpill(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // Constants
    %weight_table = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // CMX Buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <1171200> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <1150464> -> !ParentOutputDistributed
    %wt_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !WeightsTableDistributed
    %wt_cmx1 = VPURT.DeclareBuffer <CMX_NN> <1024> -> !WeightsTableDistributed
    %weights_cmx = VPURT.DeclareBuffer <CMX_NN> <3584> -> !WeightsDistributed

    // DDR buffers
    %wt_ddr = VPURT.DeclareBuffer <DDR> <0> -> !WeightsTable_DDR
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights : !Weights_DDR) outputs(%weights_cmx : !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weight table
    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table : !WeightsTable_DDR) outputs(%wt_cmx : !WeightsTableDistributed) -> !WeightsTableDistributed
    }
    VPURT.Task waits(%bar2 : !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64, spillId = 1 : i64} inputs(%wt_cmx : !WeightsTableDistributed) outputs(%wt_ddr : !WeightsTable_DDR) -> !WeightsTable_DDR
    }
    VPURT.Task waits(%bar3 : !VPURT.Barrier) updates(%bar4 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64, spillId = 1 : i64} inputs(%wt_ddr : !WeightsTable_DDR) outputs(%wt_cmx1 : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster task
    VPURT.Task waits(%bar4, %bar1, %bar0 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
        %49 = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6233 : i64,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>
                }
        input(%parent_input_cmx : !ParentInputDistributed)
        weights(%weights_cmx : !WeightsDistributed)
        weight_table(%wt_cmx1 : !WeightsTableDistributed)
        parent_input(%parent_input_cmx : !ParentInputDistributed)
        parent_output(%parent_out_cmx : !ParentOutputDistributed)
        outputs(%parent_out_cmx : !ParentOutputDistributed) -> !ParentOutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 31], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 63], outStart = [0, 0, 32],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WT0:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [32, 1, 1, 4]>]
    //CHECK:    [[WT1:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.SubView<[32, 0, 0, 0], [32, 1, 1, 4]>]

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[WT_SPILL_BUFF_1_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_SPILL_BUFF_1_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_SPILL_BUFF_0_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_SPILL_BUFF_0_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_SPILL_BUFF_2_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_SPILL_BUFF_2_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1024> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_SPILL_BUFF_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<32x1x1x4xsi32, @DDR>
    //CHECK:    [[WT_SPILL_BUFF_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <512> -> memref<32x1x1x4xsi32, @DDR>
    //CHECK:    [[WT_DDR_0:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<32x1x1x4xsi32, @DDR>
    //CHECK:    [[WT_DDR_1:%.+]] = VPURT.DeclareBuffer <DDR> <512> -> memref<32x1x1x4xsi32, @DDR>

    //CHECK:    VPURT.Task updates([[BAR_2]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64} inputs([[WT0]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_SPILL_BUFF_0_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    VPURT.Task updates([[BAR_2]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 1 : i64} inputs([[WT1]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_SPILL_BUFF_0_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_2]] : !VPURT.Barrier) updates([[BAR_3]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64, spillId = 1 : i64} inputs([[WT_SPILL_BUFF_1_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           outputs([[WT_DDR_0]] : memref<32x1x1x4xsi32, @DDR>) -> memref<32x1x1x4xsi32, @DDR>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_2]] : !VPURT.Barrier) updates([[BAR_3]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 1 : i64, spillId = 1 : i64} inputs([[WT_SPILL_BUFF_1_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           outputs([[WT_DDR_1]] : memref<32x1x1x4xsi32, @DDR>) -> memref<32x1x1x4xsi32, @DDR>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_3]] : !VPURT.Barrier) updates([[BAR_4]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64, spillId = 1 : i64} inputs([[WT_SPILL_BUFF_DDR_0]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_SPILL_BUFF_2_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    VPURT.Task waits([[BAR_3]] : !VPURT.Barrier) updates([[BAR_4]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 1 : i64, spillId = 1 : i64} inputs([[WT_SPILL_BUFF_DDR_1]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_SPILL_BUFF_2_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    }

    //CHECK:    VPURT.Task waits([[BAR_4]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weight_table([[WT_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }

    //CHECK:     VPURT.Task waits([[BAR_4]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 32 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weight_table([[WT_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    20x1x64x1x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[7, 1, 64, 1, 4], [7, 1, 64, 1, 4], [6, 1, 64, 1, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]],
    memory_shapes = [[7, 1, 64, 1, 4], [7, 1, 64, 1, 4], [6, 1, 64, 1, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]]
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    20x1x16x1x4xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[7, 1, 16, 1, 4], [7, 1, 16, 1, 4], [6, 1, 16, 1, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]],
    memory_shapes = [[7, 1, 16, 1, 4], [7, 1, 16, 1, 4], [6, 1, 16, 1, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    20x16x64x1x1xf16, #GNHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[7, 16, 64, 1, 1], [7, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]],
    memory_shapes = [[7, 16, 64, 1, 1], [7, 16, 64, 1, 1], [6, 16, 64, 1, 1]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    20x16x1x1x4xsi32, #NCDHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [3, 1, 1, 1, 1],
    num_clusters = 3 : i64, uniform_distributed_segments,
    compute_shapes = [[7, 16, 1, 1, 4], [7, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]],
    memory_shapes = [[7, 16, 1, 1, 4], [7, 16, 1, 1, 4], [6, 16, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0, 0], [7, 0, 0, 0, 0], [14, 0, 0, 0, 0]]
}>

!Input_DDR =  memref<20x1x64x1x4xf16, #GNHWC, @DDR>
!Output_DDR = memref<20x1x16x1x4xf16, #GNHWC, @DDR>
!Weights_DDR = memref<20x16x64x1x1xf16, #GNHWC, @DDR>
!WeightsTable_DDR = memref<20x16x1x1x4xsi32>

//CHECK-LABEL: @DoNotOptimizeDMAsFor5DTensorUnevenSplitSizes
func.func @DoNotOptimizeDMAsFor5DTensorUnevenSplitSizes(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //Constants
    %wt = const.Declare !WeightsTable_DDR = dense<1> : tensor<20x16x1x1x4xsi32>

    // CMX Buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <1170944> -> !ParentInputDistributed
    %weights_cmx = VPURT.DeclareBuffer <CMX_NN> <10240> -> !WeightsDistributed
    %wt_cmx = VPURT.DeclareBuffer <CMX_NN> <1174528> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <24576> -> !ParentOutputDistributed

    // DDR buffers
    %weights = VPURT.DeclareBuffer <DDR> <0> -> !Weights_DDR
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
        %97 = VPUIP.NNDMA {port = 0 : i64} inputs(%wt : !WeightsTable_DDR) outputs(%wt_cmx : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
        %97 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights : !Weights_DDR) outputs(%weights_cmx : !WeightsDistributed) -> !WeightsDistributed
    }

    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
        %97 = VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in : !Input_DDR) outputs(%parent_input_cmx : !ParentInputDistributed) -> !ParentInputDistributed
    }

    VPURT.Task waits(%bar2, %bar1, %bar0 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%bar3 : !VPURT.Barrier) {
        %97 = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
        input(%parent_input_cmx : !ParentInputDistributed)
        weights(%weights_cmx : !WeightsDistributed)
        weight_table(%wt_cmx : !WeightsTableDistributed)
        parent_input(%parent_input_cmx : !ParentInputDistributed)
        parent_output(%parent_out_cmx : !ParentOutputDistributed)
        outputs(%parent_out_cmx : !ParentOutputDistributed)
        -> !ParentOutputDistributed variants : {
            DPUTask {cluster_id = 0 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 0, 15], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 0, 15], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 2 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 0, 15], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
    }

    VPURT.Task waits(%bar3 : !VPURT.Barrier) {
        %97 = VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx : !ParentOutputDistributed) outputs(%parent_out : !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[WT0:%.+]] = const.Declare memref<7x16x1x1x4xsi32> = dense<1> : tensor<20x16x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0, 0], [7, 16, 1, 1, 4]>]
    //CHECK:    [[WT1:%.+]] = const.Declare memref<7x16x1x1x4xsi32> = dense<1> : tensor<20x16x1x1x4xsi32>, [#const.SubView<[7, 0, 0, 0, 0], [7, 16, 1, 1, 4]>]
    //CHECK:    [[WT2:%.+]] = const.Declare memref<6x16x1x1x4xsi32> = dense<1> : tensor<20x16x1x1x4xsi32>, [#const.SubView<[14, 0, 0, 0, 0], [6, 16, 1, 1, 4]>]

    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NNDMA {port = 0 : i64} inputs([[WT0]] : memref<7x16x1x1x4xsi32>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<7x16x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<7x16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    }
    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NNDMA {port = 1 : i64} inputs([[WT1]] : memref<7x16x1x1x4xsi32>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<7x16x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<7x16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    }
    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NNDMA {port = 0 : i64, split_candidate} inputs([[WT2]] : memref<6x16x1x1x4xsi32>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<6x16x1x1x4xsi32, [@CMX_NN, 2]>) -> memref<6x16x1x1x4xsi32, [@CMX_NN, 2]>
    //CHECK:    }

    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:       kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<7x1x64x1x4xf16, #GNHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<7x16x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weight_table({{[^:]+}} : memref<7x16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<7x1x64x1x4xf16, #GNHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 0]>) -> memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 0]> variants : {
    //CHECK:        DPUTask {cluster_id = 0 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    //CHECK-SAME:       outEnd = [3, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:       kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<7x1x64x1x4xf16, #GNHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<7x16x64x1x1xf16, #GNHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weight_table({{[^:]+}} : memref<7x16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<7x1x64x1x4xf16, #GNHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 1]>) -> memref<7x1x16x1x4xf16, #GNHWC, [@CMX_NN, 1]> variants : {
    //CHECK:        DPUTask {cluster_id = 1 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    //CHECK-SAME:       outEnd = [3, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
    //CHECK:    VPURT.Task
    //CHECK:      VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:       kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<6x1x64x1x4xf16, #GNHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<6x16x64x1x1xf16, #GNHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:   weight_table({{[^:]+}} : memref<6x16x1x1x4xsi32, [@CMX_NN, 2]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<6x1x64x1x4xf16, #GNHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : memref<6x1x16x1x4xf16, #GNHWC, [@CMX_NN, 2]>)
    //CHECK-SAME:   outputs({{[^:]+}} : memref<6x1x16x1x4xf16, #GNHWC, [@CMX_NN, 2]>) -> memref<6x1x16x1x4xf16, #GNHWC, [@CMX_NN, 2]> variants : {
    //CHECK:        DPUTask {cluster_id = 2 : i64, inEnd = [3, 0, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
    //CHECK-SAME:       outEnd = [3, 0, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x128x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 128, 4, 1], [1, 128, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x64x4x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 32, 4, 1], [1, 32, 4, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]],
    memory_shapes = [[1, 64, 4, 1], [1, 64, 4, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x128x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 128, 1, 1], [32, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64,
    alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]],
    memory_shapes = [[32, 1, 1, 4], [32, 1, 1, 4]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0]]
}>

!Input_DDR = memref<1x128x4x1xf16, #NHWC, @DDR>
!Output_DDR = memref<1x64x4x1xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x128x1x1xf16, #NHWC, @DDR>
!WeightsTable_DDR = memref<64x1x1x4xsi32, @DDR>

//CHECK-LABEL: @DoNotOptimizeDMAsForDifferentValuesPerCluster
func.func @DoNotOptimizeDMAsForDifferentValuesPerCluster(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // Constants
    %weight_table = const.Declare !WeightsTable_DDR = dense<"0x00000000FFFFFF000000803F0080144200240000FFFFFF000000803F0080264400480000FFFFFF000000803F00C0CB43006C0000FFFFFF000000803F00A03A4500900000FFFFFF000000803F00E00A4300B40000FFFFFF000000803F00A0E04400D80000FFFFFF000000803F00A0204500FC0000FFFFFF000000803F00808E4400200100FFFFFF000000803F00404C4500440100FFFFFF000000803F0040C14400681100FFFFFF000000803F00000643008C0100FFFFFF000000803F00405E4500B00100FFFFFF000000803F00C0054500D40100FFFFFF000000803F0060964400F81100FFFFFF000000803F00E02745001C0200FFFFFF000000803F0040C74400400200FFFFFF000000803F00C0114100640200FFFFFF000000803F00C07D4500882200FFFFFF000000803F00A0CA4400AC0200FFFFFF000000803F00E04A4400D00200FFFFFF000000803F0040384400F40200FFFFFF000000803F00205C4500180300FFFFFF000000803F00405945003C0300FFFFFF000000803F00207B4500600300FFFFFF000000803F00A0C64400840300FFFFFF000000803F00A0604500A80300FFFFFF000000803F0040704500CC0300FFFFFF000000803F0080384500F00300FFFFFF000000803F00A0F04400140400FFFFFF000000803F00408C4400380400FFFFFF000000803F00603B45005C0400FFFFFF000000803F0020B74300800400FFFFFF000000803F0060C54200A40400FFFFFF000000803F00A0964400C80400FFFFFF000000803F0060054500EC0400FFFFFF000000803F00E0854400100500FFFFFF000000803F00C04A4500340500FFFFFF000000803F00E0254300580500FFFFFF000000803F00004643007C0500FFFFFF000000803F00206A4300A00500FFFFFF000000803F0040204400C40500FFFFFF000000803F00402D4500E80500FFFFFF000000803F00002D44000C0600FFFFFF000000803F0080CE4200300600FFFFFF000000803F0040094400540600FFFFFF000000803F00A0584500780600FFFFFF000000803F00A03745009C0600FFFFFF000000803F00E03E4400000000FFFFFF000000803F00207D4500240000FFFFFF000000803F0080B84400480000FFFFFF000000803F00402644006C0000FFFFFF000000803F00E0114500900000FFFFFF000000803F0040524500B40000FFFFFF000000803F00A0394500D80000FFFFFF000000803F00A0EC4400FC0000FFFFFF000000803F0000514500200100FFFFFF000000803F0080474500440100FFFFFF000000803F0020F34400681100FFFFFF000000803F00402845008C0100FFFFFF000000803F0060E54400B00100FFFFFF000000803F00A04D4500D40100FFFFFF000000803F0040424300F81100FFFFFF000000803F00E05542001C0200FFFFFF000000803F00C03145"> : tensor<64x1x1x4xsi32>
    %weights = const.Declare !Weights_DDR = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]

    // CMX Buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <1171200> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <1150464> -> !ParentOutputDistributed
    %wt_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !WeightsTableDistributed
    %weights_cmx = VPURT.DeclareBuffer <CMX_NN> <3584> -> !WeightsDistributed

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weight table
    VPURT.Task updates(%bar1 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weight_table : !WeightsTable_DDR) outputs(%wt_cmx : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar2 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA {port = 0 : i64} inputs(%weights : !Weights_DDR) outputs(%weights_cmx : !WeightsDistributed) -> !WeightsDistributed
    }

    // Cluster task
    VPURT.Task waits(%bar2, %bar1, %bar0 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
        %49 = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 6233 : i64,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>
                }
        input(%parent_input_cmx : !ParentInputDistributed)
        weights(%weights_cmx : !WeightsDistributed)
        weight_table(%wt_cmx : !WeightsTableDistributed)
        parent_input(%parent_input_cmx : !ParentInputDistributed)
        parent_output(%parent_out_cmx : !ParentOutputDistributed)
        outputs(%parent_out_cmx : !ParentOutputDistributed) -> !ParentOutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 31], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [0, 3, 127], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 3, 63], outStart = [0, 0, 32],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA {port = 0 : i64} inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:      [[WT_DDR_0:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR>
    //CHECK-SAME: [#const.SubView<[0, 0, 0, 0], [32, 1, 1, 4]>]
    //CHECK:      [[WT_DDR_1:%.+]] = const.Declare memref<32x1x1x4xsi32, @DDR>
    //CHECK-SAME: [#const.SubView<[32, 0, 0, 0], [32, 1, 1, 4]>]

    //CHECK:    [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[WT_BUFF_0_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_BUFF_0_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK:    [[WT_BUFF_1_CMX_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK:    [[WT_BUFF_1_CMX_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 0 : i64} inputs([[WT_DDR_0]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_BUFF_1_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task updates([[BAR_1]] : !VPURT.Barrier) {
    //CHECK:        VPUIP.NNDMA {port = 1 : i64} inputs([[WT_DDR_1]] : memref<32x1x1x4xsi32, @DDR>)
    //CHECK-SAME:           outputs([[WT_BUFF_1_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>) -> memref<32x1x1x4xsi32, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR_2]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 0 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   weight_table([[WT_BUFF_0_CMX_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }

    //CHECK:     VPURT.Task waits([[BAR_2]], [[BAR_1]], [[BAR_0]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) {
    //CHECK:        VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    //CHECK-SAME:           out_channel_offset = 32 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    //CHECK-SAME:   input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weights({{[^:]+}} : memref<32x128x1x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   weight_table([[WT_BUFF_0_CMX_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_input({{[^:]+}} : memref<1x128x4x1xf16, #NHWC, [@CMX_NN, 1]>)
    //CHECK-SAME:   parent_output({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) output_ITI_buff({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 0],
    //CHECK:        ]>) outputs({{[^:]+}} : !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:        ]>) -> !VPUIP.ITIBuffer<
    //CHECK:        1x64x4x1xf16, #NHWC, [@CMX_NN, 1],
    //CHECK:    variants : {
    //CHECK:      } PPE : {
    //CHECK:      }
    //CHECK:    }
}
