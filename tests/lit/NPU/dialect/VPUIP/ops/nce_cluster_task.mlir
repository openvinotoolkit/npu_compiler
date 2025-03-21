//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @ParsePrintClusterTask(%arg0: memref<1x32x16x16xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>
                = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %out_buff_cmx = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    %weight_table = const.Declare memref<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %weight_table_buff_cmx = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    %weight_table_cmx = VPUIP.Copy inputs(%weight_table : memref<64x1x1x4xsi32>) outputs(%weight_table_buff_cmx : memref<64x1x1x4xsi32, @CMX_NN>) -> memref<64x1x1x4xsi32, @CMX_NN>

    %t1, %r1 = async.execute
                -> !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
                    attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %0 = VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }  input(%arg0 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
                weights(%weights : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
                weight_table(%weight_table_cmx : memref<64x1x1x4xsi32, @CMX_NN>)
                parent_input(%arg0 : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
                parent_output(%out_buff_cmx : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
                outputs(%out_buff_cmx : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
                    -> memref<1x64x14x14xf16, #NHWC, @CMX_NN> variants :  {
                DPUTask {
                    outStart = [0, 0, 0], outEnd = [31, 15, 15],
                    mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
                } PPE :  {
                }
        async.yield %0 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    }


    %0 = async.await %r1 : !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
    return %0 : memref<1x64x14x14xf16, #NHWC, @CMX_NN>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @CMX_NN>

    // CHECK:        [[OUT_BUFF_CMX:%.+]] = memref.alloc() : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:        [[WEIGHT_TABLE:%.+]] = const.Declare memref<64x1x1x4xsi32>

    // CHECK:        [[WEIGHT_TABLE_BUFF_CMX:%.+]] = memref.alloc() : memref<64x1x1x4xsi32, @CMX_NN>
    // CHECK:        [[WEIGHT_TABLE_CMX:%.+]] = VPUIP.Copy

    // CHECK:        [[TOKEN:%.+]], [[RESULTS:%.+]] = async.execute -> !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>> attributes {VPUIP.executor = @DPU, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:        [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:           input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:           weights([[WEIGHTS]] : memref<64x32x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:           weight_table([[WEIGHT_TABLE_CMX]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:           parent_input({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:           parent_output([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:           outputs([[OUT_BUFF_CMX]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>) -> memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:            async.yield [[NCE_TASK]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:        }
    // CHECK:        [[WAIT_RESULTS:%.+]] = async.await [[RESULTS]] : !async.value<memref<1x64x14x14xf16, #NHWC, @CMX_NN>>
    // CHECK:        return [[WAIT_RESULTS]] : memref<1x64x14x14xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    strides = [1, 1],
    num_clusters = 4
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x64x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4
}>

!Input_DDR = memref<1x32x16x16xf16, #NHWC, @DDR>
!Weights_DDR = memref<64x32x3x3xf16, #NHWC, @DDR>
!Output_DDR = memref<1x64x16x16xf16, #NHWC, @DDR>

!WeightsTableStub = memref<64x1x1x4xsi32>

func.func @ParsePrintDistributedBuffer(%input: !Input_DDR) -> !Output_DDR {
    %weights = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %input_cmx = VPURT.AllocDistributed -> !InputDistributed
    %t0 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        VPUIP.NNDMA inputs(%input: !Input_DDR) outputs(%input_cmx: !InputDistributed) -> !InputDistributed

        async.yield
    }

    %weights_cmx = VPURT.AllocDistributed -> !WeightsDistributed
    %t1 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        VPUIP.NNDMA inputs(%weights: !Weights_DDR) outputs(%weights_cmx: !WeightsDistributed) -> !WeightsDistributed

        async.yield
    }

    %output_buff_cmx = VPURT.AllocDistributed -> !OutputDistributed
    %weights_table_cmx = VPURT.AllocDistributed -> !WeightsTableDistributed
    %t2 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        %1 = const.Declare !WeightsTableStub = dense<10> : tensor<64x1x1x4xsi32>
        VPUIP.NNDMA inputs(%1: !WeightsTableStub) outputs(%weights_table_cmx: !WeightsTableDistributed) -> !WeightsTableDistributed

        async.yield
    }

    %t3 = async.execute [%t0, %t1, %t2]
                attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
            %0 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<CONV>
                }  input(%input_cmx : !InputDistributed)
                    weights(%weights_cmx : !WeightsDistributed)
                    weight_table(%weights_table_cmx : !WeightsTableDistributed)
                    parent_input(%input_cmx : !InputDistributed)
                    parent_output(%output_buff_cmx : !OutputDistributed)
                    outputs(%output_buff_cmx : !OutputDistributed)
                        -> !OutputDistributed variants :  {
                    DPUTask {
                        outStart = [0, 0, 0], outEnd = [31, 15, 15],
                        mpe_mode = #VPU.mpe_mode<VECTOR_FP16>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                    }
                    } PPE :  {
                    }

            async.yield
    }

    %output = memref.alloc() : !Output_DDR
    %t4 = async.execute
            attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
        VPUIP.NNDMA inputs(%output_buff_cmx: !OutputDistributed) outputs(%output: !Output_DDR) -> !Output_DDR

        async.yield
    }

    return %output: !Output_DDR

    // CHECK:        [[CST:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK:        [[INPUT_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN,
    // CHECK-SAME:                           {mode = "OVERLAPPED", num_tiles = [1, 1, 4, 1], kernel = [3, 3],
    // CHECK-SAME:                           pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, strides = [1, 1], num_clusters = 4 : i64}>
    // CHECK:        [[TOKEN:%.+]] = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:              VPUIP.NNDMA inputs({{[^:]+}} : memref<1x32x16x16xf16, #NHWC, @DDR>
    // CHECK-SAME:                          outputs([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    // CHECK:              async.yield
    // CHECK:        }

    // CHECK:        [[WEIGHTS_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:        [[TOKEN_0:%.+]] = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:              VPUIP.NNDMA inputs([[CST]] : memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK-SAME:                          outputs([[WEIGHTS_CMX]] : !VPUIP.DistributedBuffer
    // CHECK:              async.yield
    // CHECK:        }

    // CHECK:        [[OUTPUT_BUFF_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:        [[WEIGHTS_TABLE_CMX:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:        [[TOKEN_1:%.+]] = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:              [[CST_4:%.+]] = const.Declare memref<64x1x1x4xsi32>
    // CHECK:              VPUIP.NNDMA inputs([[CST_4]] : memref<64x1x1x4xsi32>
    // CHECK-SAME:                          outputs([[WEIGHTS_TABLE_CMX]] : !VPUIP.DistributedBuffer
    // CHECK:              async.yield
    // CHECK:        }

    // CHECK:        {{[^:]+}} = async.execute [[[TOKEN]], [[TOKEN_0]], [[TOKEN_1]]] attributes {VPUIP.executor = @DPU, VPUIP.num_units = 4 : i64, "async-deps-index" = 3 : i64} {
    // CHECK:                VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:               input([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:               weights([[WEIGHTS_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:               weight_table([[WEIGHTS_TABLE_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:               parent_input([[INPUT_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:               parent_output([[OUTPUT_BUFF_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:               outputs([[OUTPUT_BUFF_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:                   -> !VPUIP.DistributedBuffer
    // CHECK-SAME:           variants :  {
    // CHECK:                  DPUTask {mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, outEnd = [31, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    // CHECK:                } PPE :  {
    // CHECK:                }
    // CHECK:              async.yield
    // CHECK:        }

    // CHECK:        [[OUTPUT:%.+]] = memref.alloc() : memref<1x64x16x16xf16, #NHWC, @DDR>
    // CHECK:        {{[^:]+}} = async.execute attributes {VPUIP.executor = @DMA_NN, VPUIP.num_units = 1 : i64, "async-deps-index" = 0 : i64} {
    // CHECK:              VPUIP.NNDMA inputs([[OUTPUT_BUFF_CMX]] : !VPUIP.DistributedBuffer
    // CHECK-SAME:                          outputs([[OUTPUT]] : memref<1x64x16x16xf16, #NHWC, @DDR>
    // CHECK:              async.yield
    // CHECK:        }
    // CHECK:        return [[OUTPUT]] : memref<1x64x16x16xf16, #NHWC, @DDR>
}
