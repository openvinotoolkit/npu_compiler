//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --unroll-distributed-ops --canonicalize --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    32x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!SpilledOutput_DDR = memref<1x32x32x32xf16, {order = #NHWC}, @DDR>

!Weights_DDR = memref<32x16x1x1xf16, {order = #NHWC}, @DDR>
!NewWeightsTables_DDR = memref<32x1x1x1xf32, @DDR>

// CHECK-LABEL: @UnrollDistributedOps
module @UnrollDistributedOps {
    func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
        // Barriers
        %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        %weights_cst = const.Declare !Weights_DDR =
            dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table_cst = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<32x1x1x1xf32>
        %bias_table_cst = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<32x1x1x1xf32>

        // DDR buffers
        %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
        %spilled_out = VPURT.DeclareBuffer <DDR> <0> -> !SpilledOutput_DDR

        // CMX buffers
        %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
        %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
        %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
        %bias_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
        %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

        // Upload input
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
        }

        // Upload weights
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
        }

        // Upload scale table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !NewWeightsTables_DDR) outputs(%scale_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
        }

        // Upload bias table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !NewWeightsTables_DDR) outputs(%bias_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
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
                        weight_table_scale(%scale_table : !NewWeightsTablesDistributed)
                        weight_table_bias(%bias_table : !NewWeightsTablesDistributed)
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
                        } PPE :  {
                        }
        }

        // Copyback output
        VPURT.Task waits(%bar1: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
        }

        return %output: !Output_DDR

        //CHECK:    [[SCALE_TABLE1_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>
        //CHECK:    [[SCALE_TABLE2_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>
        //CHECK:    [[BIAS_TABLE1_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>
        //CHECK:    [[BIAS_TABLE2_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>

        //CHECK-DAG:    [[WEIGHTS1_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR>
        //CHECK-DAG:    [[WEIGHTS2_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR>

        //CHECK:        [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK:        [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        //CHECK:        [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
        //CHECK:        [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:        [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

        //CHECK-DAG:    [[INPUT_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

        //CHECK-DAG:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK-DAG:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
        // Upload 1st part of weights
        //CHECK-DAG:    [[WEIGHTS1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        // Upload 2nd part of weights
        //CHECK-DAG:    [[WEIGHTS2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

        //CHECK:    [[SCALE_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        //CHECK:    [[SCALE_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>
        // Upload 1st part of scale table
        //CHECK:    [[SCALE_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        // Upload 2nd part of scale table
        //CHECK:    [[SCALE_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>

        //CHECK:    [[BIAS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        //CHECK:    [[BIAS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>
        // Upload 1st part of bias table
        //CHECK:    [[BIAS_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        // Upload 2nd part of bias table
        //CHECK:    [[BIAS_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>


        //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> !VPUIP.ITIBuffer<
        //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
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

        //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> !VPUIP.ITIBuffer<
        //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
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

        // Upload input
        //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NNDMA
        //CHECK-SAME:       inputs([[INPUT_DDR]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)
        //CHECK-SAME:       outputs([[INPUT_CMX_COPY]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>)
        //CHECK:        }

        // 1st subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
        //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
        //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE1_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE1_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 0]>)
        //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK:                parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
        //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }

        // 2nd subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1],
        //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
        //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE2_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE2_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 1]>)
        //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK:                parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
        //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!ScaleTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x1xf8E5M2, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!BiasTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!SpilledOutput_DDR = memref<1x32x32x32xf16, {order = #NHWC}, @DDR>

!Weights_DDR = memref<32x16x1x1xf16, {order = #NHWC}, @DDR>
!ScaleTable_DDR = memref<32x1x1x1xf8E5M2, @DDR>
!BiasTable_DDR = memref<32x1x1x1xf32, @DDR>

// CHECK-LABEL: @UnrollDistributedOpsWithF8E5M2Scales
module @UnrollDistributedOpsWithF8E5M2Scales {
    func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
        // Barriers
        %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        %weights_cst = const.Declare !Weights_DDR =
            dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table_cst = const.Declare !ScaleTable_DDR = dense<1.0> : tensor<32x1x1x1xf8E5M2>
        %bias_table_cst = const.Declare !BiasTable_DDR = dense<2.0> : tensor<32x1x1x1xf32>

        // DDR buffers
        %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
        %spilled_out = VPURT.DeclareBuffer <DDR> <0> -> !SpilledOutput_DDR

        // CMX buffers
        %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
        %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
        %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !ScaleTableDistributed
        %bias_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !BiasTableDistributed
        %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

        // Upload input
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
        }

        // Upload weights
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
        }

        // Upload scale table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !ScaleTable_DDR) outputs(%scale_table: !ScaleTableDistributed) -> !ScaleTableDistributed
        }

        // Upload bias table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !BiasTable_DDR) outputs(%bias_table: !BiasTableDistributed) -> !BiasTableDistributed
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
                        weight_table_scale(%scale_table : !ScaleTableDistributed)
                        weight_table_bias(%bias_table : !BiasTableDistributed)
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
                        } PPE :  {
                        }
        }

        // Copyback output
        VPURT.Task waits(%bar1: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
        }

        return %output: !Output_DDR

        //CHECK:    [[BIAS_TABLE1_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>
        //CHECK:    [[BIAS_TABLE2_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR>
        //CHECK:    [[SCALE_TABLE1_CST:%.+]] = const.Declare memref<16x1x1x1xf8E5M2, @DDR>
        //CHECK:    [[SCALE_TABLE2_CST:%.+]] = const.Declare memref<16x1x1x1xf8E5M2, @DDR>

        //CHECK-DAG:    [[WEIGHTS1_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR>
        //CHECK-DAG:    [[WEIGHTS2_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR>

        //CHECK:        [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK:        [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        //CHECK:        [[INPUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
        //CHECK:        [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:        [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

        //CHECK-DAG:    [[INPUT_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

        //CHECK-DAG:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK-DAG:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
        // Upload 1st part of weights
        //CHECK-DAG:    [[WEIGHTS1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        // Upload 2nd part of weights
        //CHECK-DAG:    [[WEIGHTS2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

        //CHECK:    [[SCALE_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf8E5M2, [@CMX_NN, 0]>
        //CHECK:    [[SCALE_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf8E5M2, [@CMX_NN, 1]>
        // Upload 1st part of scale table
        //CHECK:    [[SCALE_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf8E5M2, [@CMX_NN, 0]>
        // Upload 2nd part of scale table
        //CHECK:    [[SCALE_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf8E5M2, [@CMX_NN, 1]>

        //CHECK:    [[BIAS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        //CHECK:    [[BIAS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>
        // Upload 1st part of bias table
        //CHECK:    [[BIAS_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
        // Upload 2nd part of bias table
        //CHECK:    [[BIAS_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>


        //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> !VPUIP.ITIBuffer<
        //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
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

        //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> !VPUIP.ITIBuffer<
        //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
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

        // Upload input
        //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NNDMA
        //CHECK-SAME:       inputs([[INPUT_DDR]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)
        //CHECK-SAME:       outputs([[INPUT_CMX_COPY]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>)
        //CHECK:        }

        // 1st subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
        //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE1_CMX]] : memref<16x1x1x1xf8E5M2, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE1_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 0]>)
        //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK:                parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
        //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }

        // 2nd subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
        //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE2_CMX]] : memref<16x1x1x1xf8E5M2, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE2_CMX]] : memref<16x1x1x1xf32, [@CMX_NN, 1]>)
        //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK:                parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
        //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
        //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    32x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!SpilledOutput_DDR = memref<1x32x32x32xf16, {order = #NHWC}, @DDR>

!Weights_DDR = memref<32x16x1x1xf16, {order = #NHWC}, @DDR>
!NewWeightsTables_DDR = memref<32x1x1x1xf32, @DDR>

// CHECK-LABEL: @UnrollDistributedOpsDuplicatedScaleAndBiasTables
module @UnrollDistributedOpsDuplicatedScaleAndBiasTables {
    func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
        // Barriers
        %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        %weights_cst = const.Declare !Weights_DDR =
            dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
        %scale_table_cst = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<32x1x1x1xf32>
        %bias_table_cst = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<32x1x1x1xf32>

        // DDR buffers
        %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
        %spilled_out = VPURT.DeclareBuffer <DDR> <0> -> !SpilledOutput_DDR

        // CMX buffers
        %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
        %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
        %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
        %bias_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
        %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

        // Upload input
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
        }

        // Upload weights
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
        }

        // Upload scale table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !NewWeightsTables_DDR) outputs(%scale_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
        }

        // Upload bias table
        VPURT.Task updates(%bar0: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !NewWeightsTables_DDR) outputs(%bias_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
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
                        weight_table_scale(%scale_table : !NewWeightsTablesDistributed)
                        weight_table_bias(%bias_table : !NewWeightsTablesDistributed)
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
                        } PPE :  {
                        }
        }

        // Copyback output
        VPURT.Task waits(%bar1: !VPURT.Barrier) {
            VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
        }

        return %output: !Output_DDR

        //CHECK:    [[SCALE_TABLE_CST:%.+]] = const.Declare memref<32x1x1x1xf32, @DDR>
        //CHECK:    [[BIAS_TABLE_CST:%.+]] = const.Declare memref<32x1x1x1xf32, @DDR>

        //CHECK-DAG:    [[WEIGHTS_CST:%.+]] = const.Declare memref<32x16x1x1xf16, {order = #NHWC}, @DDR>

        //CHECK:        [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        //CHECK:        [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        //CHECK:        [[INPUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @DDR>
        //CHECK:        [[INPUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> <16> -> memref<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @DDR>
        //CHECK:        [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:        [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
        //CHECK:        [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:        [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

        //CHECK-DAG:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK-DAG:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
        // Upload weights
        //CHECK-DAG:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <32768> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

        //CHECK:    [[SCALE_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
        //CHECK:    [[SCALE_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
        // Upload scale tables
        //CHECK:    [[SCALE_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33280> -> !VPUIP.DistributedBuffer<32x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

        //CHECK:    [[BIAS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
        //CHECK:    [[BIAS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
        // Upload bias tables
        //CHECK:    [[BIAS_TABLE_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33280> -> !VPUIP.DistributedBuffer<32x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>


        //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
        //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

        // Upload input
        //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NNDMA
        //CHECK-SAME:       inputs([[INPUT1_DDR]] : memref<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @DDR>)
        //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK:        }

        //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NNDMA
        //CHECK-SAME:       inputs([[INPUT2_DDR]] : memref<1x8x32x32xf16, {order = #NHWC, strides = [16384, 1, 512, 16]}, @DDR>)
        //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK:        }

        // 1st subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
        //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE1_CMX]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE1_CMX]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
        //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
        //CHECK:                parent_output([[OUT1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:                outputs([[OUT1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }

        // 2nd subtask
        //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
        //CHECK:          VPUIP.NCEClusterTask <{
        //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
        //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_scale([[SCALE_TABLE2_CMX]] : memref<32x1x1x1xf32, [@CMX_NN, 1]>)
        //CHECK-SAME:           weight_table_bias([[BIAS_TABLE2_CMX]] : memref<32x1x1x1xf32, [@CMX_NN, 1]>)
        //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x8x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
        //CHECK:                parent_output([[OUT2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
        //CHECK:                outputs([[OUT2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
        //CHECK:            variants :  {
        //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
        //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        //CHECK:          } PPE :  {
        //CHECK:          }
        //CHECK:        }
    }
}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x5x5xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 5], [1, 16, 2, 5]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>


!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x1x22x22xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 11, 22], [1, 1, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 1, 12, 22], [1, 1, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    16x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x10x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
}>

!InputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>
!NewWeightsTables_DDR = memref<16x1x1x1xf32, @DDR>

!Output_DDR = memref<1x16x10x10xf16, {order = #NHWC}, @DDR>

//CHECK-LABEL: @UnrollNceSoHSEPInterpolateOverlappedTwoClusters
func.func @UnrollNceSoHSEPInterpolateOverlappedTwoClusters() -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %seTable_cst = const.Declare memref<1x1x22x22xi32, {order = #NHWC}, @DDR> = dense<[[[
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
        ]]]> : tensor<1x1x22x22xi32, {order = #NHWC}>
    %seTable_CMX = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, {order = #NHWC}, @DDR>) outputs(%seTable_CMX : !InputSETableDistributed) -> !InputSETableDistributed
    }

    %scale_table_cst = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    %bias_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    // Upload scale table

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !NewWeightsTables_DDR) outputs(%scale_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    // Upload bias table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !NewWeightsTables_DDR) outputs(%bias_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }>   input(%parent_input_cmx : !InputDistributed)
                input_storage_element_table(%seTable_CMX: !InputSETableDistributed)
                weights(%weights : !WeightsDistributed)
                weight_table_scale(%scale_table : !NewWeightsTablesDistributed)
                weight_table_bias(%bias_table : !NewWeightsTablesDistributed)
                parent_input(%parent_input_cmx : !InputDistributed)
                parent_input_storage_element_table(%seTable_CMX: !InputSETableDistributed)
                parent_output(%parent_out_cmx : !OutputDistributed)
                outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %parent_out: !Output_DDR

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, {order = #NHWC}, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, {order = #NHWC}, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x5x5xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 3, 5], [1, 16, 2, 5]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 3, 0]],
    memory_shapes = [[1, 16, 3, 5], [1, 16, 3, 5]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]
}>


!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x1x22x22xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 11, 22], [1, 1, 11, 22]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 11, 0]],
    memory_shapes = [[1, 1, 12, 22], [1, 1, 12, 22]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 10, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x4x4xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[16, 16, 4, 4], [16, 16, 4, 4]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x10x10xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]],
    memory_shapes = [[1, 16, 5, 10], [1, 16, 5, 10]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 5, 0]]
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    16x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>
!NewWeightsTables_DDR = memref<16x1x1x1xf32, @DDR>

!InputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>

!Output_DDR = memref<1x16x10x10xf16, {order = #NHWC}, @DDR>

//CHECK-LABEL: @UnrollNceSoHSEPInterpolateOverlappedTwoClustersAndTwoUsers
func.func @UnrollNceSoHSEPInterpolateOverlappedTwoClustersAndTwoUsers() -> (!Output_DDR, !Output_DDR) {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %seTable_cst = const.Declare memref<1x1x22x22xi32, {order = #NHWC}, @DDR> = dense<[[[
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [5121, 5121, 5121, 5121, 5121, 6145, 6145, 6145, 6145, 7169, 7169, 7169, 7169, 8193, 8193, 8193, 8193, 9217, 9217, 9217, 9217, 9217],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337],
        [10241, 10241, 10241, 10241, 10241, 11265, 11265, 11265, 11265, 12289, 12289, 12289, 12289, 13313, 13313, 13313, 13313, 14337, 14337, 14337, 14337, 14337]
        ]]]> : tensor<1x1x22x22xi32, {order = #NHWC}>

    // First NCE user
    %seTable_CMX_0 = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, {order = #NHWC}, @DDR>) outputs(%seTable_CMX_0 : !InputSETableDistributed) -> !InputSETableDistributed
    }

    %scale_table_cst_0 = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst_0 = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table_0 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    %bias_table_0 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed

    // Upload scale table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst_0: !NewWeightsTables_DDR) outputs(%scale_table_0: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    // Upload bias table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst_0: !NewWeightsTables_DDR) outputs(%bias_table_0: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    %parent_out_0 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    %parent_input_cmx_0 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %weights_0 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %parent_out_cmx_0 = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed

    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }>   input(%parent_input_cmx_0 : !InputDistributed)
                input_storage_element_table(%seTable_CMX_0: !InputSETableDistributed)
                weights(%weights_0 : !WeightsDistributed)
                weight_table_scale(%scale_table_0 : !NewWeightsTablesDistributed)
                weight_table_bias(%bias_table_0 : !NewWeightsTablesDistributed)
                parent_input(%parent_input_cmx_0 : !InputDistributed)
                parent_input_storage_element_table(%seTable_CMX_0: !InputSETableDistributed)
                parent_output(%parent_out_cmx_0 : !OutputDistributed)
                outputs(%parent_out_cmx_0 : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx_0: !OutputDistributed) outputs(%parent_out_0: !Output_DDR) -> !Output_DDR
    }

    // Second NCE user
    %seTable_CMX_1 = VPURT.DeclareBuffer <CMX_NN> <4160> -> !InputSETableDistributed
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%seTable_cst : memref<1x1x22x22xi32, {order = #NHWC}, @DDR>) outputs(%seTable_CMX_1 : !InputSETableDistributed) -> !InputSETableDistributed
    }
    %scale_table_cst_1 = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst_1 = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table_1 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    %bias_table_1 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed

    // Upload scale table
    VPURT.Task updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst_1: !NewWeightsTables_DDR) outputs(%scale_table_1: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    // Upload bias table
    VPURT.Task updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst_1: !NewWeightsTables_DDR) outputs(%bias_table_1: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    %parent_out_1 = VPURT.DeclareBuffer <NetworkOutput> [10240] <0> -> !Output_DDR

    %parent_input_cmx_1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_input_se_table_1 = VPURT.DeclareBuffer <CMX_NN> <33442> -> !InputSETableDistributed
    %weights_1 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <26400> -> !WeightsDistributed
    %parent_out_cmx_1 = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed

    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [4, 4],
                    kernel_strides = [2, 2],
                    task_type = #VPUIP.nce_task_type<CONV>
            }>   input(%parent_input_cmx_1 : !InputDistributed)
                input_storage_element_table(%seTable_CMX_1 : !InputSETableDistributed)
                weights(%weights_1 : !WeightsDistributed)
                weight_table_scale(%scale_table_1 : !NewWeightsTablesDistributed)
                weight_table_bias(%bias_table_1 : !NewWeightsTablesDistributed)
                parent_input(%parent_input_cmx_1 : !InputDistributed)
                parent_input_storage_element_table(%seTable_CMX_1: !InputSETableDistributed)
                parent_output(%parent_out_cmx_1 : !OutputDistributed)
                outputs(%parent_out_cmx_1 : !OutputDistributed)
                        -> !OutputDistributed
                variants :  {
                    DPUTask {
                        cluster_id = 0 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    DPUTask {
                        cluster_id = 1 : i64,
                        inEnd = [21, 11, 15], inStart = [0, 0, 0],
                        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                        outEnd = [9, 4, 15], outStart = [0, 0, 0],
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                } PPE :  {
                }
    }

    VPURT.Task waits(%bar4: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%parent_out_cmx_1: !OutputDistributed) outputs(%parent_out_1: !Output_DDR) -> !Output_DDR
    }

    return %parent_out_0, %parent_out_1: !Output_DDR, !Output_DDR

    // This test is to check the same SETable constant is used by two different NCE ops
    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, {order = #NHWC}, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>

    // CHECK{LITERAL}:  const.Declare memref<1x1x12x22xi32, {order = #NHWC}, @DDR> = dense<[[[
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [0, 0, 0, 0, 0, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 3072, 3072, 3072, 3072, 4096, 4096, 4096, 4096, 4096]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [5120, 5120, 5120, 5120, 5120, 6144, 6144, 6144, 6144, 7168, 7168, 7168, 7168, 8192, 8192, 8192, 8192, 9216, 9216, 9216, 9216, 9216]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK-SAME:      [10240, 10240, 10240, 10240, 10240, 11264, 11264, 11264, 11264, 12288, 12288, 12288, 12288, 13312, 13312, 13312, 13312, 14336, 14336, 14336, 14336, 14336]
    // CHECK{LITERAL}:  ]]]> : tensor<1x1x12x22xi32, {order = #NHWC}>
}

// -----

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


!InputSETableDistributed = !VPUIP.DistributedBuffer<
    1x16x33x33xi32, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 3, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 3
}>


!WeightsDistributed = !VPUIP.DistributedBuffer<
    16x16x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    16x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>
!NewWeightsTables_DDR = memref<16x1x1x1xf32, @DDR>

!Input_DDR = memref<1x16x33x33xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x33x33xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<16x16x3x3xf16, {order = #NHWC}, @DDR>

!InputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!InputSETableStub_CMX = memref<1x16x33x33xi32, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x3x3xf16, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @UnrollNceSoHOutputOverlapped
// CHECK-SAME: ([[ARG_0:%.+]]: memref<1x16x33x33xf16, {order = #NHWC}, @DDR>, [[ARG_1:%.+]]: memref<1x16x33x33xf16, {order = #NHWC}, @DDR>)
func.func @UnrollNceSoHOutputOverlapped(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <13728> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <26400> -> !WeightsDistributed

    %parent_input_se_table = VPURT.DeclareBuffer <CMX_NN> <33442> -> !InputSETableDistributed

    %scale_table_cst = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    %bias_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed

    // Upload scale table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !NewWeightsTables_DDR) outputs(%scale_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    // Upload bias table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !NewWeightsTables_DDR) outputs(%bias_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                    kernel_size = [3, 3],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
            }>   input(%parent_input_cmx : !InputDistributed)
                input_storage_element_table(%parent_input_se_table : !InputSETableDistributed)
                weights(%weights : !WeightsDistributed)
                weight_table_scale(%scale_table : !NewWeightsTablesDistributed)
                weight_table_bias(%bias_table : !NewWeightsTablesDistributed)
                parent_input(%parent_input_cmx : !InputDistributed)
                parent_input_storage_element_table(%parent_input_se_table : !InputSETableDistributed)
                parent_output(%parent_out_cmx : !OutputDistributed)
                outputs(%parent_out_cmx : !OutputDistributed)
                        -> !OutputDistributed
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
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:    [[BIAS_CST:%.+]] = const.Declare memref<16x1x1x1xf32, @DDR> = dense<2.000000e+00> : tensor<16x1x1x1xf32>
    //CHECK:    [[SCALE_CST:%.+]]  = const.Declare memref<16x1x1x1xf32, @DDR> = dense<1.000000e+00> : tensor<16x1x1x1xf32>
    //CHECK:    [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x3x3xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:    [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:    [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x12x33xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <10560> -> memref<1x16x13x33xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[IN3_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <22176> -> memref<1x16x12x33xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[OUT1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x13x33xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <12672> -> memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK:    [[OUT3_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <23232> -> memref<1x16x11x33xf16, {order = #NHWC}, @DDR>

    //CHECK:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:    [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN3_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 2]>

    //CHECK:    [[OUT1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13728> -> memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[OUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <13728> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[OUT3_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <13728> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>

    //CHECK:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13728> ->
    //CHECK-SAME: !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK-NEXT: inwardHaloRegions = [
    //CHECK-NEXT:   #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 33], offset = [0, 0, 11, 0], cluster_id = 0 : i64>
    //CHECK-NEXT:  ]
    //CHECK-NOT:  outwardHaloRegions

    //CHECK:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <13728> ->
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

    //CHECK:    [[OUT3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <13728> ->
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

    //CHECK:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <26400> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <26400> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <26400> -> memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <26400> -> !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:    [[IN_SE_TABLE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33442> -> memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[IN_SE_TABLE_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33442> -> memref<1x16x13x33xi32, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[IN_SE_TABLE_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <33442> -> memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 2]>

    //CHECK-NOT:  outwardHaloRegions



    //CHECK:    [[SCALE1:%.+]]     = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:    [[SCALE2:%.+]]     = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>
    //CHECK:    [[SCALE3:%.+]]     = VPURT.DeclareBuffer <CMX_NN> [2] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 2]>
    //CHECK:    [[SCALE_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <33280> -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>
    //CHECK:    [[BIAS1:%.+]]      = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 0]>
    //CHECK:    [[BIAS2:%.+]]      = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 1]>
    //CHECK:    [[BIAS3:%.+]]      = VPURT.DeclareBuffer <CMX_NN> [2] <33280> -> memref<16x1x1x1xf32, [@CMX_NN, 2]>
    //CHECK:    [[BIAS_COPY:%.+]]  = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <33280> -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:     VPURT.Task updates([[BAR0]] : !VPURT.Barrier)  {
    //CHECK:       VPUIP.NNDMA
    //CHECK-SAME:    inputs([[SCALE_CST]] : memref<16x1x1x1xf32, @DDR>)
    //CHECK-SAME:    outputs([[SCALE_COPY]] : !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:     }
    //CHECK:     VPURT.Task updates([[BAR0]] : !VPURT.Barrier)  {
    //CHECK:       VPUIP.NNDMA
    //CHECK-SAME:    inputs([[BIAS_CST]] : memref<16x1x1x1xf32, @DDR>)
    //CHECK-SAME:    outputs([[BIAS_COPY]] : !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:     }

    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x12x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x13x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN3_DDR]] : memref<1x16x12x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN3_CMX_COPY]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x3x3xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_1]] : memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_scale([[SCALE1]] : memref<16x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_bias([[BIAS1]] : memref<16x1x1x1xf32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_1]] : memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 10, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>}

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_2]] : memref<1x16x13x33xi32, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table_scale([[SCALE2]] : memref<16x1x1x1xf32, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table_bias([[BIAS2]] : memref<16x1x1x1xf32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_2]] : memref<1x16x13x33xi32, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            output_ITI_buff([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:            outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 21, 15], outStart = [0, 11, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    //CHECK-SAME:           kernel_size = [3, 3], kernel_strides = [1, 1],
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN3_CMX]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           input_storage_element_table([[IN_SE_TABLE_3]] : memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x3x3xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table_scale([[SCALE3]] : memref<16x1x1x1xf32, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table_bias([[BIAS3]] : memref<16x1x1x1xf32, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x12x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_input_storage_element_table([[IN_SE_TABLE_3]] : memref<1x16x12x33xi32, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:                output_ITI_buff([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:                outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 32, 15], outStart = [0, 22, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>}

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT1_CMX_COPY]] : memref<1x16x13x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT1_DDR]] : memref<1x16x13x33xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT2_CMX_COPY]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:       outputs([[OUT2_DDR]] : memref<1x16x11x33xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT3_CMX_COPY]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:       outputs([[OUT3_DDR]] : memref<1x16x11x33xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    //CHECK:    return [[ARG_1]] : memref<1x16x33x33xf16, {order = #NHWC}, @DDR>
}

// -----

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

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    16x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>
!NewWeightsTables_DDR = memref<16x1x1x1xf32, @DDR>

!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x48x32x32xf16, {order = #NHWC}, @DDR>
!Weights_DDR = memref<48x16x1x1xf16, {order = #NHWC}, @DDR>

!InputStub_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x48x32x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<48x16x1x1xf16, {order = #NHWC}, @CMX_NN>

//CHECK-LABEL: @UnrollNceSOK
func.func @UnrollNceSOK(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare !Weights_DDR =
        dense<1.0> : tensor<48x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %scale_table_cst = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed
    %bias_table = VPURT.DeclareBuffer <CMX_NN> <33344> -> !NewWeightsTablesDistributed

    // Upload scale table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: !NewWeightsTables_DDR) outputs(%scale_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    // Upload bias table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: !NewWeightsTables_DDR) outputs(%bias_table: !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <32768> -> !ParentOutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <131072> -> !WeightsDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
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
                    weight_table_scale(%scale_table : !NewWeightsTablesDistributed)
                    weight_table_bias(%bias_table : !NewWeightsTablesDistributed)
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
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:  [[WEIGHTS1_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:  [[WEIGHTS2_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]
    //CHECK-DAG:  [[WEIGHTS3_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}, @DDR> = dense<1.000000e+00> : tensor<48x16x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 16, 1, 1]>, #const.Reorder<#NHWC>]

    //CHECK:      [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:      [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:      [[IN_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
    //CHECK:      [[OUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x48x32x32xf16, {order = #NHWC}, @DDR>

    //CHECK:      [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:      [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:      [[IN3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:      [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <0> -> !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>

    //CHECK:      [[OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:      [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
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

    //CHECK:      [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
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

    //CHECK:      [[OUT3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <32768> ->
    //CHECK-SAME:   !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]
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

    //CHECK:      [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:      [[WEIGHTS1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:      [[WEIGHTS2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:      [[WEIGHTS3_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <131072> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weight
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of weight
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS3_CST]] : memref<16x16x1x1xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[WEIGHTS3_CMX_COPY]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:           output_ITI_buff([[OUT2_CMX]], [[OUT3_CMX]]
    //CHECK:           outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:           variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN3_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:           output_ITI_buff([[OUT1_CMX]], [[OUT2_CMX]]
    //CHECK:           outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:        variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 47], outStart = [0, 0, 32],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.+]] : memref<1x48x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x48x32x32xf16, {order = #NHWC}, @DDR>)
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

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    16x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 3,
    uniform_distributed_segments
}>
!NewWeightsTables_DDR = memref<16x1x1x1xf32, @DDR>

!Input_DDR = memref<1x16x33x33xf16, {order = #NHWC}>
!Output_DDR = memref<1x16x33x33xf16, {order = #NHWC}>
!Weights_DDR = memref<16x16x1x1xf16, {order = #NHWC}>

!InputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x33xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>

//CHECK-LABEL: @UnrollNceHKSwitch
func.func @UnrollNceHKSwitch(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<16x16x1x1xf16, {order = #NHWC}> =
        dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <11616> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46464> -> !WeightsDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input_cmx: !InputDistributed) -> !InputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR) outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    %scale_table_cst = const.Declare memref<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
    %bias_table_cst = const.Declare memref<16x1x1x1xf32> = dense<2.0> : tensor<16x1x1x1xf32>

    %scale_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <47000> -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>
    %bias_table = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <47064> -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>

    // Upload scale table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst: memref<16x1x1x1xf32>) outputs(%scale_table: !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>
    }
    // Upload bias table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst: memref<16x1x1x1xf32>) outputs(%bias_table: !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input_cmx : !InputDistributed)
                    weights(%weights : !WeightsDistributed)
                    weight_table_scale(%scale_table : !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>)
                    weight_table_bias(%bias_table : !VPUIP.DistributedBuffer<16x1x1x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3, uniform_distributed_segments}>)
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
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !OutputDistributed) outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK:     [[WEIGHTS_CST:%.+]] = const.Declare memref<16x16x1x1xf16, {order = #NHWC}>

    //CHECK:     [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:     [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK:     [[IN1_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK:     [[IN2_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <11616> -> memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK:     [[IN3_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <23232> -> memref<1x16x11x33xf16, {order = #NHWC}, @DDR>

    //CHECK:     [[OUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x16x33x33xf16, {order = #NHWC}>

    //CHECK:     [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:     [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:     [[IN3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:     [[IN1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:     [[IN2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:     [[IN3_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>


    //CHECK:     [[OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <11616> -> memref<1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 0]>

    //CHECK:     [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 0]
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

    //CHECK:     [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 1]
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

    //CHECK:     [[OUT3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <11616> ->
    //CHECK-SAME:  !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 2]
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

    //CHECK:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <46464> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <46464> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK:    [[WEIGHTS3_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <46464> -> memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>
    //CHECK:    [[WEIGHTS_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2] <46464> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>


    // Upload 1st part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN1_DDR]] : memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN1_CMX_COPY]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN2_DDR]] : memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN2_CMX_COPY]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 3rd part of input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN3_DDR]] : memref<1x16x11x33xf16, {order = #NHWC}, @DDR>
    //CHECK-SAME:       outputs([[IN3_CMX_COPY]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK:        }

    // Upload weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_CST]] : memref<16x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS_CMX_COPY]] : !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments}>)
    //CHECK:        }

    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN1_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_output([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                output_ITI_buff([[OUT2_CMX]], [[OUT3_CMX]]
    //CHECK:                outputs([[OUT1_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:           variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 10, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN2_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_output([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            output_ITI_buff([[OUT1_CMX]], [[OUT3_CMX]]
    //CHECK:            outputs([[OUT2_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:        variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 21, 15], outStart = [0, 11, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // 3rd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN3_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weights([[WEIGHTS3_CMX]] : memref<16x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN3_CMX]] : memref<1x16x11x33xf16, {order = #NHWC}, [@CMX_NN, 2]>)
    //CHECK-SAME:           parent_output([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:                output_ITI_buff([[OUT1_CMX]], [[OUT2_CMX]]
    //CHECK:                outputs([[OUT3_CMX]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 2]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 2 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [32, 32, 15], outStart = [0, 22, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    // CHECK:          VPUIP.NNDMA
    // CHECK-SAME:       inputs([[OUT_CMX]] : memref<1x16x33x33xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x16x33x33xf16, {order = #NHWC}>)
    // CHECK:        }

}

// -----


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

!NewWeightsTablesDistributed1 = !VPUIP.DistributedBuffer<
    32x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    uniform_distributed_segments
}>
!NewWeightsTablesDistributed2 = !VPUIP.DistributedBuffer<
    32x1x1x1xf32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2,
    uniform_distributed_segments
}>

!NewWeightsTables_DDR = memref<32x1x1x1xf32, @DDR>


!Input_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x16x32x32xf16, {order = #NHWC}, @DDR>
!SpilledOutput_DDR = memref<1x32x32x32xf16, {order = #NHWC}, @DDR>

!Weights1_DDR = memref<32x16x1x1xf16, {order = #NHWC}, @DDR>

!InputStub1_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub1_CMX = memref<1x32x32x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub1_CMX = memref<32x16x1x1xf16, {order = #NHWC}, @CMX_NN>

!Weights2_DDR = memref<32x32x1x1xf16, {order = #NHWC}, @DDR>

!InputStub2_CMX = memref<1x32x32x32xf16, {order = #NHWC}, @CMX_NN>
!OutputStub2_CMX = memref<1x16x32x32xf16, {order = #NHWC}, @CMX_NN>
!WeightsStub2_CMX = memref<32x32x1x1xf16, {order = #NHWC}, @CMX_NN>

func.func @UnrollNCESequence(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights1_cst = const.Declare !Weights1_DDR =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %weights2_cst = const.Declare !Weights2_DDR =
        dense<1.0> : tensor<32x32x1x1xf16>, [#const.Reorder<#NHWC>]

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR
    %spilled_out = VPURT.DeclareBuffer <DDR> <0> -> !SpilledOutput_DDR

    // CMX buffers/ 1st task
    %parent_input1_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed_1
    %weights1 = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed_1
    %parent_out1_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed_1

    // CMX buffers/ 2nd task
    %spilled_input_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !SpilledInputDistributed
    %parent_input2_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentInputDistributed_2
    %weights2 = VPURT.DeclareBuffer <CMX_NN> <99072> -> !WeightsDistributed_2
    %parent_out2_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentOutputDistributed_2

    // Add scale/bias tables for first task
    %scale_table_cst_1 = const.Declare !NewWeightsTables_DDR = dense<1.0> : tensor<32x1x1x1xf32>
    %bias_table_cst_1 = const.Declare !NewWeightsTables_DDR = dense<2.0> : tensor<32x1x1x1xf32>

    %scale_table_1 = VPURT.DeclareBuffer <CMX_NN> <33280> -> !NewWeightsTablesDistributed1
    %bias_table_1 = VPURT.DeclareBuffer <CMX_NN> <33408> -> !NewWeightsTablesDistributed1

    // Add scale/bias tables for second task
    %scale_table_cst_2 = const.Declare memref<32x1x1x1xf32, @DDR> = dense<1.0> : tensor<32x1x1x1xf32>
    %bias_table_cst_2 = const.Declare memref<32x1x1x1xf32, @DDR> = dense<2.0> : tensor<32x1x1x1xf32>

    %scale_table_2 = VPURT.DeclareBuffer <CMX_NN> <130048> -> !NewWeightsTablesDistributed2
    %bias_table_2 = VPURT.DeclareBuffer <CMX_NN> <130176> -> !NewWeightsTablesDistributed2
    // Upload scale/bias tables for first task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst_1: !NewWeightsTables_DDR) outputs(%scale_table_1: !NewWeightsTablesDistributed1) -> !NewWeightsTablesDistributed1
    }
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst_1: !NewWeightsTables_DDR) outputs(%bias_table_1: !NewWeightsTablesDistributed1) -> !NewWeightsTablesDistributed1
    }
    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR) outputs(%parent_input1_cmx: !ParentInputDistributed_1) -> !ParentInputDistributed_1
    }

    // Upload weights/ 1st task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights1_cst: !Weights1_DDR) outputs(%weights1: !WeightsDistributed_1) -> !WeightsDistributed_1
    }

    // Cluster tiling/ 1st task
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input1_cmx : !ParentInputDistributed_1)
                    weights(%weights1 : !WeightsDistributed_1)
                    weight_table_scale(%scale_table_1 : !NewWeightsTablesDistributed1)
                    weight_table_bias(%bias_table_1 : !NewWeightsTablesDistributed1)
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
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out1_cmx: !ParentOutputDistributed_1) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
    }

    // Upload input
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%spilled_out: !SpilledOutput_DDR) outputs(%spilled_input_cmx: !SpilledInputDistributed) -> !SpilledInputDistributed
    }

    // Upload weights/ 2nd task
    VPURT.Task waits(%bar2: !VPURT.Barrier) updates(%bar3: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights2_cst: !Weights2_DDR) outputs(%weights2: !WeightsDistributed_2) -> !WeightsDistributed_2
    }
    // Upload scale/bias tables for second task
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%scale_table_cst_2: memref<32x1x1x1xf32, @DDR>) outputs(%scale_table_2: !NewWeightsTablesDistributed2)
            -> !NewWeightsTablesDistributed2
    }
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%bias_table_cst_2: memref<32x1x1x1xf32, @DDR>) outputs(%bias_table_2: !NewWeightsTablesDistributed2)
            -> !NewWeightsTablesDistributed2
    }


    // Cluster tiling/ 2nd task
    VPURT.Task waits(%bar3: !VPURT.Barrier) updates(%bar4: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }>  input(%parent_input2_cmx : !ParentInputDistributed_2)
                    weights(%weights2 : !WeightsDistributed_2)
                    weight_table_scale(%scale_table_2 : !NewWeightsTablesDistributed2)
                    weight_table_bias(%bias_table_2 : !NewWeightsTablesDistributed2)
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
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out2_cmx: !ParentOutputDistributed_2) outputs(%spilled_out: !SpilledOutput_DDR) -> !SpilledOutput_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS1_CST_2ND_TASK:%.+]] = const.Declare memref<16x32x1x1xf16, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[WEIGHTS2_CST_2ND_TASK:%.+]] = const.Declare memref<16x32x1x1xf16, {order = #NHWC}, @DDR>

    //CHECK:        [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[SPILLED_OUTPUT_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x32x32x32xf16, {order = #NHWC}, @DDR>

    //CHECK-DAG:    [[SPILLED_OUTPUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[INPUT2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>

    //CHECK:        [[IN1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK:        [[IN2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33536> -> memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK-DAG:    [[WEIGHTS1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    // Upload 1st part of weights/ 2nd task
    //CHECK-DAG:    [[WEIGHTS1_CMX_COPY_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <99072> -> memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // Upload 2nd part of weights/ 2nd task
    //CHECK-DAG:    [[WEIGHTS2_CMX_COPY_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <99072> -> memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

    //CHECK:    [[OUT1_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
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

    //CHECK:    [[OUT2_CMX_2ND_TASK:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !VPUIP.ITIBuffer<
    //CHECK-NEXT:   1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
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

    // Upload input, weights for 1st cluster task
    // Process 1st cluster task

    // Spill 1st output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) updates([[BAR2]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[SPILLED_OUTPUT_CMX]] : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[SPILLED_OUTPUT_DDR]] : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>)
    //CHECK:        }

    // Upload input
    //CHECK:        VPURT.Task waits([[BAR2]] : !VPURT.Barrier) updates([[BAR3]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[SPILLED_OUTPUT_DDR]] : memref<1x32x32x32xf16, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[INPUT2_CMX_COPY]] : !VPUIP.DistributedBuffer<1x32x32x32xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments}>)
    //CHECK:        }

   // 2nd task/ 1st subtask
    //CHECK:        VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX_2ND_TASK]] : memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN1_CMX_2ND_TASK]] : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:                parent_output([[OUT1_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:                outputs([[OUT1_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 0]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task/ 2nd subtask
    //CHECK:        VPURT.Task waits([[BAR3]] : !VPURT.Barrier) updates([[BAR4]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1]
    //CHECK-SAME:       }> input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX_2ND_TASK]] : memref<16x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table_scale
    //CHECK-SAME:           weight_table_bias
    //CHECK-SAME:           parent_input([[IN2_CMX_2ND_TASK]] : memref<1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:                parent_output([[OUT2_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:                outputs([[OUT2_CMX_2ND_TASK]] : !VPUIP.ITIBuffer<
    //CHECK-NEXT:               1x32x32x32xf16, {order = #NHWC}, [@CMX_NN, 1]
    //CHECK:            variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }
}
