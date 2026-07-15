//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU3720

!qElemType = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType1 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:0, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
!qElemType5 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
!qElemType6 = !quant.uniform<u8<0:254>:f16:1, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

// CHECK: !qElemType = !quant.uniform<u8<0:254>:f16:1, {6.3053641732283461E-4:127,6.4447357898622052E-4:127,5.8824434055118114E-4:127,5.1855853223425191E-4:127,6.8580447219488186E-4:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
// CHECK: !qElemType1 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127,1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
// CHECK: !qElemType2 = !quant.uniform<u8<0:254>:f16:0, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
// CHECK: !qElemType3 = !quant.uniform<u8<0:254>:f16:0, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>
// CHECK: !qElemType4 = !quant.uniform<u8<0:254>:f16:1, {1.000000e+00:127,2.000000e+00:127,3.000000e+00:127,4.000000e+00:127,5.000000e+00:127,6.000000e+00:127,7.000000e+00:127,8.000000e+00:127,9.000000e+00:127,1.000000e+01:127,1.100000e+01:127,1.200000e+01,1.300000e+01:127,1.400000e+01:127,1.500000e+01:127,1.600000e+01:127}>
// CHECK: !qElemType5 = !quant.uniform<u8<0:254>:f16:1, {1.700000e+01:127,1.800000e+01:127,1.900000e+01:127,2.000000e+01:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127,1.000000e+00:127}>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32x!qElemType1, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1x!qElemType3, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!Input_DDR = memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
!Output_DDR = memref<1x32x32x32x!qElemType1, {order = #NHWC}, @DDR>

!Weights_DDR = memref<32x16x1x1x!qElemType3, {order = #NHWC}>
!WeightsTable = memref<32x1x1x4xsi32>

!InputStub_CMX = memref<1x16x32x32x!qElemType, {order = #NHWC}, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32x!qElemType1, {order = #NHWC}, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1x!qElemType3, {order = #NHWC}, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, @CMX_NN>

func.func @UnrollNCE(%input: !Input_DDR, %output: !Output_DDR) -> !Output_DDR {
    // Barriers
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %weights_cst = const.Declare memref<32x16x1x1x!qElemType3, {order = #NHWC}> =
        dense<1.0> : tensor<32x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]
    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    // DDR buffers
    %parent_in = VPURT.DeclareBuffer <NetworkInput> <0> -> !Input_DDR
    %parent_out = VPURT.DeclareBuffer <NetworkOutput> <0> -> !Output_DDR

    // CMX buffers
    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

    // Upload input
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_in: !Input_DDR)
                    outputs(%parent_input_cmx: !ParentInputDistributed) -> !ParentInputDistributed
    }

    // Upload weights
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_cst: !Weights_DDR)
                    outputs(%weights: !WeightsDistributed) -> !WeightsDistributed
    }

    // Upload weights table
    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%weights_table_cst: !WeightsTable)
                    outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    // Cluster tiling
    VPURT.Task waits(%bar0: !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
        %1 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
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
                        outStart = [0, 0, 0], outEnd = [31, 31, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 0 : i64
                    }
                    DPUTask {
                        outStart = [0, 0, 16], outEnd = [31, 31, 31],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        cluster_id = 1 : i64
                    }
                    } PPE :  {
                    }
    }

    // Copyback output
    VPURT.Task waits(%bar1: !VPURT.Barrier) {
        VPUIP.NNDMA <{port = 0 : i64}> inputs(%parent_out_cmx: !ParentOutputDistributed)
                    outputs(%parent_out: !Output_DDR) -> !Output_DDR
    }

    return %output: !Output_DDR

    //CHECK-DAG:    [[WEIGHTS_TABLE1_CST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    //CHECK-DAG:    [[WEIGHTS1_CST:%.+]] = const.Declare memref<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    //CHECK-DAG:    [[WEIGHTS2_CST:%.+]] = const.Declare memref<16x16x1x1x!qElemType3, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 1, 1]>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType3>, #const.Reorder<#NHWC>]

    //CHECK:        [[BAR0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:        [[BAR1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    //CHECK-DAG:    [[IN_DDR:%.+]] = VPURT.DeclareBuffer <NetworkInput> <0> -> memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[OUT_DDR:%.+]] = VPURT.DeclareBuffer <NetworkOutput> <0> -> memref<1x32x32x32x!qElemType1, {order = #NHWC}, @DDR>
    //CHECK-DAG:    [[PARENT_IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-DAG:    [[IN1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x32x32x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[IN2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x16x32x32x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[IN_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK-DAG:    [[WEIGHTS1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<16x16x1x1x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<16x16x1x1x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[WEIGHTS_TABLE1_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 0]>
    //CHECK-DAG:    [[WEIGHTS_TABLE2_CMX_COPY:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <33280> -> memref<16x1x1x4xsi32, [@CMX_NN, 1]>
    //CHECK-DAG:    [[PARENT_OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> <33536> -> !VPUIP.DistributedBuffer<1x32x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <33536> -> memref<1x32x32x32x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>
    //CHECK-DAG:    [[OUT1_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType4, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>
    //CHECK-DAG:    [[OUT2_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <33536> -> !VPUIP.DistributedBuffer<1x16x32x32x!qElemType5, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // Upload input
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[IN_DDR]] : memref<1x16x32x32x!qElemType, {order = #NHWC}, @DDR>)
    //CHECK-SAME:       outputs([[IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK:        }

    // Upload 1st part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS1_CST]] : memref<16x16x1x1x!qElemType2, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS1_CMX_COPY]] : memref<16x16x1x1x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[WEIGHTS2_CST]] : memref<16x16x1x1x!qElemType3, {order = #NHWC}>)
    //CHECK-SAME:       outputs([[WEIGHTS2_CMX_COPY]] : memref<16x16x1x1x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK:        }

    // Upload 1st part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE1_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE1_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK:        }

    // Upload 2nd part of weights table
    //CHECK:        VPURT.Task updates([[BAR0]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA <{port = 1 : i64}>
    //CHECK-SAME:       inputs([[WEIGHTS_TABLE2_CST]] : memref<16x1x1x4xsi32>)
    //CHECK-SAME:       outputs([[WEIGHTS_TABLE2_CMX_COPY]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK:        }


    // 1st task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 0 : i64,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN1_CMX]] : memref<1x16x32x32x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weights([[WEIGHTS1_CMX]] : memref<16x16x1x1x!qElemType2, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE1_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 0]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT1_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType4, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 15], outStart = [0, 0, 0],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // 2nd task
    //CHECK:        VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NCEClusterTask <{
    //CHECK-SAME:           kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    //CHECK-SAME:           kernel_size = [1, 1], kernel_strides = [1, 1], out_channel_offset = 16 : i64,
    //CHECK-SAME:           task_type = #VPUIP.nce_task_type<CONV>
    //CHECK-SAME:       }> input([[IN2_CMX]] : memref<1x16x32x32x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weights([[WEIGHTS2_CMX]] : memref<16x16x1x1x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>)
    //CHECK-SAME:           weight_table([[WEIGHTS_TABLE2_CMX]] : memref<16x1x1x4xsi32, [@CMX_NN, 1]>)
    //CHECK-SAME:           parent_input([[PARENT_IN_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>)
    //CHECK-SAME:           parent_output([[PARENT_OUT_CMX]] : !VPUIP.DistributedBuffer<1x32x32x32x!qElemType1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:           outputs([[OUT2_CMX]] : !VPUIP.DistributedBuffer<1x16x32x32x!qElemType5, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>)
    //CHECK-SAME:       variants :  {
    //CHECK:                DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 31, 31], outStart = [0, 0, 16],
    //CHECK-SAME:               pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    //CHECK:          } PPE :  {
    //CHECK:          }
    //CHECK:        }

    // Copyback output
    //CHECK:        VPURT.Task waits([[BAR1]] : !VPURT.Barrier) {
    //CHECK:          VPUIP.NNDMA
    //CHECK-SAME:       inputs([[OUT_CMX:%.+]] : memref<1x32x32x32x!qElemType1, {order = #NHWC}, [@CMX_NN, 0]>)
    //CHECK-SAME:       outputs([[OUT_DDR]] : memref<1x32x32x32x!qElemType1, {order = #NHWC}, @DDR>)
    //CHECK:        }
}
