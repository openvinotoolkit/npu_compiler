//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --patch-weight-table %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTable
func.func @PatchWeightTable() ->  memref<1008x1x1x4xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>


    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<1008x1x1x4xsi32>) outputs(%weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>

    %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = #VPUIP.nce_task_type<DWCONV>} input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) weights(%weights : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=16128 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    // CHECK:       [[NDMA_OP:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableWeightsOnly
func.func @PatchWeightTableWeightsOnly() -> memref<256x1x1x1xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <428608> -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<256x1x1x1xsi32>) outputs(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>} input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>) weights(%weights : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [112, 4, 63], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    // CHECK:       [[NDMA_OP:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<256x1x1x1xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<64x16x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PatchWeightTableWithSpill
func.func @PatchWeightTableWithSpill() ->  memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]> {
    %weight_table_1 = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    %weight_table_DDR = VPURT.DeclareBuffer <DDR> <0> -> memref<1008x1x1x4xsi32, @DDR>
    %weight_table_2 = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>

    %2 = VPUIP.NNDMA inputs(%weight_table_const : memref<1008x1x1x4xsi32>) outputs(%weight_table_1 : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPUIP.NNDMA inputs(%weight_table_1 : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%weight_table_DDR : memref<1008x1x1x4xsi32, @DDR>) -> memref<1008x1x1x4xsi32, @DDR>
    %4 = VPUIP.NNDMA inputs(%weight_table_DDR : memref<1008x1x1x4xsi32, @DDR>) outputs(%weight_table_2 : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>

    %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = #VPUIP.nce_task_type<DWCONV>} input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) weights(%weights : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%weight_table_2 : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %5 : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHT_TABLE_BUF_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1008x1x1x4xsi32, @DDR>
    // CHECK:       [[WEIGHT_TABLE_BUF2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <256> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=16128 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    // CHECK:       [[NDMA_OP1:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF1]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NDMA_OP2:.+]] = VPUIP.NNDMA
    // CHECK:       [[NDMA_OP3:.+]] = VPUIP.NNDMA
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF2]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    num_clusters = 4
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
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
!WeightsTable_DDR = memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = memref<1x64x16x16xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBuffers
func.func @PatchWeightTableWithDistributedBuffers(%arg0: !Input_DDR) -> !Output_DDR {

    %buf_in = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %buf_W = VPURT.DeclareBuffer <CMX_NN> <16384> -> !WeightsDistributed
    %buf_WT = VPURT.DeclareBuffer <CMX_NN> <86016> -> !WeightsTableDistributed
    %buf_out = VPURT.DeclareBuffer <CMX_NN> <53248> -> !OutputDistributed
    %buf_out_DDR = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR

    %cst_W = const.Declare !Weights_DDR = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_WT = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %8 = VPUIP.NNDMA inputs(%arg0 : !Input_DDR) outputs(%buf_in : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %8 = VPUIP.NNDMA inputs(%cst_W : !Weights_DDR) outputs(%buf_W : !WeightsDistributed) -> !WeightsDistributed
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %8 = VPUIP.NNDMA inputs(%cst_WT : !WeightsTable_DDR) outputs(%buf_WT : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %8 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%buf_in : !InputDistributed) weights(%buf_W : !WeightsDistributed) weight_table(%buf_WT : !WeightsTableDistributed) parent_input(%buf_in : !InputDistributed) parent_output(%buf_out : !OutputDistributed) outputs(%buf_out : !OutputDistributed) -> !OutputDistributed variants :  {
          DPUTask {outEnd = [31, 15, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %8 = VPUIP.NNDMA inputs(%buf_out : !OutputDistributed) outputs(%buf_out_DDR : !Output_DDR) -> !Output_DDR
    }
    return %buf_out_DDR : !Output_DDR

    // CHECK:       [[INPUT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <53248> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK-DAG:       [[CONST_W:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK-DAG:       [[CONST_WT:%.+]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>, #const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 0, 0, 0], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[INPUT_BUF]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST_W]]
    // CHECK-SAME:          outputs([[WEIGHT_BUF]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST_WT]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[INPUT_BUF]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, num_clusters = 4 : i64}>
    // CHECK-SAME:          weights([[WEIGHT_BUF]] : !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          weight_table([[WEIGHT_TABLE_BUF]] : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[OUTPUT_BUF]]
    // CHECK-SAME:          outputs([[OUTPUT_BUF_DDR]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 4, 1],
    kernel = [3, 3],
    pads = #VPU.Padding<left = 1 , right = 1, top = 1, bottom = 1>,
    num_clusters = 4
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    64x32x3x3xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    64x1x1x4xsi32, #NHWC, @CMX_NN, {
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
!WeightsTable_DDR = memref<64x1x1x4xsi32, #NHWC, @DDR>
!Output_DDR = memref<1x64x16x16xf16, #NHWC, @DDR>

!InputStub_CMX = memref<1x32x16x16xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<64x32x3x3xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<64x1x1x4xsi32, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x64x16x16xf16, #NHWC, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBuffersWithSpilling
func.func @PatchWeightTableWithDistributedBuffersWithSpilling(%arg0: !Input_DDR) -> !Output_DDR {

    %buf_in = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %buf_W = VPURT.DeclareBuffer <CMX_NN> <16384> -> !WeightsDistributed
    %buf_WT_1 = VPURT.DeclareBuffer <CMX_NN> <86016> -> !WeightsTableDistributed
    %buf_WT_DDR = VPURT.DeclareBuffer <DDR> <0> -> !WeightsTable_DDR
    %buf_WT_2 = VPURT.DeclareBuffer <CMX_NN> <86016> -> !WeightsTableDistributed
    %buf_out = VPURT.DeclareBuffer <CMX_NN> <53248> -> !OutputDistributed
    %buf_out_DDR = VPURT.DeclareBuffer <DDR> <0> -> !Output_DDR

    %cst_W = const.Declare !Weights_DDR = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_WT = const.Declare !WeightsTable_DDR = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %5 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%5 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%arg0 : !Input_DDR) outputs(%buf_in : !InputDistributed) -> !InputDistributed
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%cst_W : !Weights_DDR) outputs(%buf_W : !WeightsDistributed) -> !WeightsDistributed
    }

    VPURT.Task waits(%5 : !VPURT.Barrier) updates(%6 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%cst_WT : !WeightsTable_DDR) outputs(%buf_WT_1 : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%buf_WT_1 : !WeightsTableDistributed) outputs(%buf_WT_DDR : !WeightsTable_DDR) -> !WeightsTable_DDR
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%buf_WT_DDR : !WeightsTable_DDR) outputs(%buf_WT_2 : !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>} input(%buf_in : !InputDistributed) weights(%buf_W : !WeightsDistributed) weight_table(%buf_WT_2 : !WeightsTableDistributed) parent_input(%buf_in : !InputDistributed) parent_output(%buf_out : !OutputDistributed) outputs(%buf_out : !OutputDistributed) -> !OutputDistributed variants :  {
          DPUTask {outEnd = [31, 15, 15], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
        }
    }

    VPURT.Task waits(%7 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
        %11 = VPUIP.NNDMA inputs(%buf_out : !OutputDistributed) outputs(%buf_out_DDR : !Output_DDR) -> !Output_DDR
    }

    return %buf_out_DDR : !Output_DDR

    // CHECK:       [[INPUT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[WEIGHT_TABLE_BUF_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<64x1x1x4xsi32, #NHWC, @DDR>
    // CHECK:       [[WEIGHT_TABLE_BUF_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> <86016> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <53248> -> !VPUIP.DistributedBuffer
    // CHECK:       [[OUTPUT_BUF_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x64x16x16xf16, #NHWC, @DDR>

    // CHECK-DAG:       [[CONST_W:%.+]] = const.Declare memref<64x32x3x3xf16, #NHWC, @DDR>
    // CHECK-DAG:       [[CONST_WT:%.+]] = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>, #const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 0, 0, 0], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs(%arg0
    // CHECK-SAME:          outputs([[INPUT_BUF]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST_W]]
    // CHECK-SAME:          outputs([[WEIGHT_BUF]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST_WT]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_1]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[WEIGHT_TABLE_BUF_1]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_DDR]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[WEIGHT_TABLE_BUF_DDR]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF_2]]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NCEClusterTask
    // CHECK-SAME:          input([[INPUT_BUF]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], kernel = [3, 3], pads = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, num_clusters = 4 : i64}>)
    // CHECK-SAME:          weights(%1 : !VPUIP.DistributedBuffer<64x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          weight_table(%4 : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:          outputs([[OUTPUT_BUF]] : !VPUIP.DistributedBuffer<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[OUTPUT_BUF]]
    // CHECK-SAME:          outputs([[OUTPUT_BUF_DDR]]

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

!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBufferWithSOHAndWeightsOnly
func.func @PatchWeightTableWithDistributedBufferWithSOHAndWeightsOnly() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <35328> -> !WeightsTableDistributed

    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {

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

    return %parent_out_cmx : !OutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <35328> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 0], weightsTableSize=256 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
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

!WeightsTable_DDR = memref<32x1x1x4xsi32, #NCHW, @DDR>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBufferWithSOKAndWeightsOnly
// For SOK, we get an incorrect weights table that will be rewritten after UnrollClusterTilingPass pass
func.func @PatchWeightTableWithDistributedBufferWithSOKAndWeightsOnly() -> !ParentOutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33536> -> !ParentOutputDistributed

    %weights_table_cst = const.Declare memref<32x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<32x1x1x4xsi32>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%weights_table_cst: !WeightsTableStub_CMX) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {

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

    return %parent_out_cmx: !ParentOutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <33280> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<32x1x1x4xsi32, @CMX_NN> = dense<1> : tensor<32x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 16], weightsTableSize=512 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
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

!WeightsTable_DDR = memref<16x1x1x4xsi32>

!InputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x16x33x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<16x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<16x1x1x4xsi32, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBufferWithSOH
func.func @PatchWeightTableWithDistributedBufferWithSOH() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <17408> -> !OutputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <34816> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <35584> -> !WeightsTableDistributed

    %weights_table_cst = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%weights_table_cst: !WeightsTable_DDR) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {


        %1 = VPUIP.NCEClusterTask {
                  kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                  kernel_size = [1, 1],
                  kernel_strides = [1, 1],
                  task_type = #VPUIP.nce_task_type<DWCONV>
              }  input(%parent_input_cmx : !InputDistributed)
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

    return %parent_out_cmx : !OutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<16x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <35584> -> !VPUIP.DistributedBuffer<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 0], weightsTableSize=256 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
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

!WeightsTable_Stub = memref<32x1x1x4xsi32, #NCHW>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBufferWithSOK
// For SOK, we get an incorrect weights table that will be rewritten after UnrollClusterTilingPass pass
func.func @PatchWeightTableWithDistributedBufferWithSOK() -> !ParentOutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33280> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <33664> -> !ParentOutputDistributed

    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%weights_table_cst: !WeightsTable_Stub) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {

        %1 = VPUIP.NCEClusterTask {
                    kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                    kernel_size = [1, 1],
                    kernel_strides = [1, 1],
                    task_type = #VPUIP.nce_task_type<DWCONV>
                }  input(%parent_input_cmx : !ParentInputDistributed)
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

    return %parent_out_cmx: !ParentOutputDistributed

    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <[[WEIGHTS_ADDR:[^>]+]]> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <33280> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]], [[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0, 16], weightsTableSize=512 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ParentInputDistributed = !VPUIP.DistributedBuffer<
    1x16x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2
}>

!ParentOutputDistributed = !VPUIP.DistributedBuffer<
    1x32x32x32xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    32x16x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTableDistributed = !VPUIP.DistributedBuffer<
    32x1x1x4xsi32, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [2, 1, 1, 1],
    num_clusters = 2
}>

!WeightsTable_Stub = memref<32x1x1x4xsi32, #NCHW>

!InputStub_CMX = memref<1x16x32x32xf16, #NHWC, @CMX_NN>
!OutputStub_CMX = memref<1x32x32x32xf16, #NHWC, @CMX_NN>
!WeightsStub_CMX = memref<32x16x1x1xf16, #NHWC, @CMX_NN>
!WeightsTableStub_CMX = memref<32x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: @PatchWeightTableWithDistributedBufferWithDuplicatedSegmentedWeights
// For SOK, we get an incorrect weights table that will be rewritten after UnrollClusterTilingPass pass
func.func @PatchWeightTableWithDistributedBufferWithDuplicatedSegmentedWeights() -> !ParentOutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %parent_input_cmx = VPURT.DeclareBuffer <CMX_NN> <0> -> !ParentInputDistributed
    %weights = VPURT.DeclareBuffer <CMX_NN> <32768> -> !WeightsDistributed
    %weights_table = VPURT.DeclareBuffer <CMX_NN> <33792> -> !WeightsTableDistributed
    %parent_out_cmx = VPURT.DeclareBuffer <CMX_NN> <34560> -> !ParentOutputDistributed

    %weights_table_cst = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    VPURT.Task updates(%bar0: !VPURT.Barrier) {
        VPUIP.NNDMA inputs(%weights_table_cst: !WeightsTable_Stub) outputs(%weights_table: !WeightsTableDistributed) -> !WeightsTableDistributed
    }

    VPURT.Task waits(%bar0: !VPURT.Barrier) {

               %1 = VPUIP.NCEClusterTask {
                         kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                         kernel_size = [1, 1],
                         kernel_strides = [1, 1],
                         task_type = #VPUIP.nce_task_type<DWCONV>
                     }  input(%parent_input_cmx : !ParentInputDistributed)
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

    return %parent_out_cmx: !ParentOutputDistributed


    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <32768> -> !VPUIP.DistributedBuffer<32x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <33792> -> !VPUIP.DistributedBuffer<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64}>
    // CHECK:       [[CONST:%.+]] = const.Declare memref<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>,
    // CHECK-SAME:          [#const.RelocateWeightsTable<weightsPtr=[32768, 33280], sparsityPtr=16777215 : i64, offsets=[0, 16], weightsTableSize=512 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]

    // CHECK:       VPURT.Task
    // CHECK-NEXT:      VPUIP.NNDMA
    // CHECK-SAME:          inputs([[CONST]]
    // CHECK-SAME:          outputs([[WEIGHT_TABLE_BUF]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>
// CHECK-LABEL: @PatchWeightTableI4WeightsOnly
func.func @PatchWeightTableI4WeightsOnly() -> memref<256x1x1x1xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<64x16x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <428608> -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<256x1x1x1xsi32>) outputs(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    %5 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}
        input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<64x16x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>)
        weight_table(%weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>)
        parent_input(%in : memref<1x16x59x227xf16, #NHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>)
        outputs(%out : memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x64x29x113xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {outEnd = [112, 4, 63], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<256x1x1x1xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1024> -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<64x16x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<256x1x1x1xsi32> = dense<1> : tensor<256x1x1x1xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=1024 : i64, weightsElemBitSize=4 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    // CHECK:       [[NDMA_OP:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<256x1x1x1xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>) -> memref<256x1x1x1xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<64x16x3x3x!qElemType, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<256x1x1x1xsi32, [@CMX_NN, 0]>)
}

// -----

func.func @PatchWeightTableWithSubViews() -> memref<64x1x1x4xsi32> {
    %cst_weight_table = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %weights = VPURT.DeclareBuffer <CMX_NN> <593920> -> !VPUIP.DistributedBuffer<512x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %weight_table = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %out = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x64x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %in = VPURT.DeclareBuffer <CMX_NN> <0> -> !VPUIP.DistributedBuffer<1x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %dma = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_weight_table : memref<64x1x1x4xsi32>) outputs(%weight_table : !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>

    %0 = VPUIP.SubView %weights [64, 0, 0, 0] [64, 448, 1, 1] : !VPUIP.DistributedBuffer<512x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}> to !VPUIP.DistributedBuffer<64x448x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [448, 1, 448, 448]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [2, 1, 1, 1]}>
    %1 = VPUIP.DistributedCast inputs(%0 : !VPUIP.DistributedBuffer<64x448x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [448, 1, 448, 448]}, @CMX_NN, {mode = "DUPLICATED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [2, 1, 1, 1]}>) -> !VPUIP.DistributedBuffer<64x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>
    %2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 716 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%in : !VPUIP.DistributedBuffer<1x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        weights(%1 : !VPUIP.DistributedBuffer<64x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        weight_table(%weight_table : !VPUIP.DistributedBuffer<64x1x1x4xsi32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, alignment = [16, 1, 1, 1]}>)
        parent_input(%in : !VPUIP.DistributedBuffer<1x448x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output(%out : !VPUIP.DistributedBuffer<1x64x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%out : !VPUIP.DistributedBuffer<1x64x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        -> !VPUIP.DistributedBuffer<1x64x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [0, 0, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    return %cst_weight_table : memref<64x1x1x4xsi32>

    // CHECK: [[CNST:%.+]] = const.Declare
    // CHECK-SAME: #const.RelocateWeightsTable<weightsPtr=[651264, 679936], sparsityPtr=16777215 : i64, offsets=[0, 32], weightsTableSize=1024 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>
    // CHECK: VPUIP.NNDMA
    // CHECK-SAME: inputs([[CNST]]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @PatchWeightTable5D
func.func @PatchWeightTable5D() ->  memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<4x1008x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<4x1008x1x1x4xsi32> = dense<1> : tensor<4x1008x1x1x4xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<4x1x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<4x1x1008x1x1xf16, #GNHWC, [@CMX_NN, 0]>

    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<4x1008x1x1x4xsi32>) outputs(%weight_table : memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>

    %5 = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%in : memref<4x1x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weights(%weights : memref<4x1008x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weight_table(%weight_table : memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%in : memref<4x1x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        parent_output(%out : memref<4x1x1008x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        outputs(%out : memref<4x1x1008x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
            -> memref<4x1x1008x1x1xf16, #GNHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<4x1008x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<4x1008x1x1x4xsi32> = dense<1> : tensor<4x1008x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[[[WEIGHTS_ADDR]]], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=64512 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    // CHECK:       [[NDMA_OP:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<4x1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<4x1008x64x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<4x1008x1x1x4xsi32, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotPatchWeightTableWhenZeroOffset
func.func @DoNotPatchWeightTableWhenZeroOffset() ->  memref<1008x1x1x4xsi32, [@CMX_NN, 0]> {
    %weight_table = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    %weights = VPURT.DeclareBuffer <CMX_NN> [0] <395136> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weight_table_const = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>

    %in = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>
    %out = VPURT.DeclareBuffer <CMX_NN> [0] <556416> -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>


    %4 = VPUIP.NNDMA inputs(%weight_table_const : memref<1008x1x1x4xsi32>) outputs(%weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>

    %5 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [7, 7], task_type = #VPUIP.nce_task_type<DWCONV>} input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) weights(%weights : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) parent_input(%in : memref<1x1008x14x14xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) outputs(%out : memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x1008x2x2xf16, #NHWC, [@CMX_NN, 0]> variants :  {
        DPUTask {outEnd = [1, 0, 1007], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE :  {
    }

    return %weight_table : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[WEIGHT_TABLE_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <540288> -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <[[WEIGHTS_ADDR:[^>]+]]> -> memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK-DAG:       [[CONST:%.+]] = const.Declare memref<1008x1x1x4xsi32> = dense<1> : tensor<1008x1x1x4xsi32>
    // CHECK:       [[NDMA_OP:.+]] = VPUIP.NNDMA inputs([[CONST]] : memref<1008x1x1x4xsi32>) outputs([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<1008x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[NCE_CLUST_TASK_OP:.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:  weights([[WEIGHTS_BUF]] : memref<1008x64x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE_BUF]] : memref<1008x1x1x4xsi32, [@CMX_NN, 0]>)
}
