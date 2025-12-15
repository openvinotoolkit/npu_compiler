//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --patch-populate-weight-table-with-shave --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
            attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @PatchSWKernelSingleCluster
// CHECK-SAME:     , [[SCALE_ARG:%.+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>
func.func @PatchSWKernelSingleCluster(%arg0: memref<1x1x2048xf32, @DDR>, %scale: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
           -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>

    %cst = const.Declare memref<416x2048x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<416x2048x1x1xf16>,
        [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %token, %bodyResults = async.execute -> !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 819 : i64, cycleEnd = 819 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x2048xf32, @DDR>) -> memref<1x1x1x2048xf32, @DDR>
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%21 : memref<1x1x1x2048xf32, @DDR>) outputs(%0 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>) -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
      async.yield %22 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    }
    %token_4, %bodyResults_5 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64, cycleBegin = 819 : i64, cycleCost = 1 : i64, cycleEnd = 820 : i64} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg2 as %arg3: memref<1x1x1x2048xf32, [@CMX_NN, 0]>)
            outputs(%1 as %arg4: memref<1x1x1x2048xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x2048xf32, [@CMX_NN, 0]>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>
      }
      async.yield %results : memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    }
    %token_6, %bodyResults_7 = async.execute [%token] -> !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 819 : i64, cycleCost = 164704 : i64, cycleEnd = 165522 : i64} {
      %21 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<416x2048x1x1x!qElemType, #NHWC>) outputs(%3 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
        -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
      async.yield %21 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    }

    %token_8, %bodyResults_9:2 = async.execute -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {
      %21 = VPUIP.SubView %2 [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %22 = VPUIP.SubView %2 [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
            outputs(%21 as %arg5: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, %22 as %arg6: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
            -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [3072, 1024]}(%arg5) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
        VPUIP.SW.Kernel.run {attrs = [429056, 1024]}(%arg6) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      }
      async.yield %results#0, %results#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    }

    %token_12, %bodyResults_13 = async.execute [%token_4, %token_6, %token_8] (%bodyResults_5 as %arg2: !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>,
                %bodyResults_9#0 as %arg3: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_9#1 as %arg4: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_7 as %arg5: !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>)
                -> !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 165522 : i64, cycleCost = 1 : i64, cycleEnd = 165523 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg2 : memref<1x1x1x2048xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, [@CMX_NN, 0]>
      %22 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%21 : memref<1x2048x1x1xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>
      %23 = VPUIP.ConcatView inputs(%arg3, %arg4 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>)
        outputs(%2 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
      %24 = VPUIP.NCEClusterTask
                {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
            input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg5 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
            weight_table(%23 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 415], outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %24 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    %result = async.await %bodyResults_13 : !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>

    return %result : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>


    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:       -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
    // CHECK-SAME:       attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {

    // CHECK:         [[SUBVIEW0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         [[SUBVIEW1:%.+]] = VPUIP.SubView [[IN_BUF]] [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:         [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:       inputs([[SCALE_ARG]] as [[INNER_ARG0:[^:]+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[INNER_ARG2:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, [[SUBVIEW1]] as [[INNER_ARG3:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:       -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [4096, 1024]}([[INNER_ARG2]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [217088, 1024]}([[INNER_ARG3]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:           async.yield [[RESULTS]]#0, [[RESULTS]]#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
            attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }

// CHECK-LABEL: @PatchSWKernelSingleClusterWithDMASpill
// CHECK-SAME:      , [[SCALE_ARG:%.+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>
func.func @PatchSWKernelSingleClusterWithDMASpill(%arg0: memref<1x1x2048xf32, @DDR>, %scale: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
           -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <DDR> <0> -> memref<208x1x1x4xsi32, @DDR>
    %6 = VPURT.DeclareBuffer <DDR> <832> -> memref<208x1x1x4xsi32, @DDR>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <2048> -> memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [0] <2880> -> memref<208x1x1x4xsi32, [@CMX_NN, 0]>

    %cst = const.Declare memref<416x2048x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<416x2048x1x1xf16>,
        [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %token, %bodyResults = async.execute -> !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 819 : i64, cycleEnd = 819 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x2048xf32, @DDR>) -> memref<1x1x1x2048xf32, @DDR>
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%21 :  memref<1x1x1x2048xf32, @DDR>) outputs(%0 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>) -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
      async.yield %22 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    }
    %token_4, %bodyResults_5 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64, cycleBegin = 819 : i64, cycleCost = 1 : i64, cycleEnd = 820 : i64} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg2 as %arg3: memref<1x1x1x2048xf32, [@CMX_NN, 0]>)
            outputs(%1 as %arg4: memref<1x1x1x2048xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x2048xf32, [@CMX_NN, 0]>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>
      }
      async.yield %results : memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    }
    %token_6, %bodyResults_7 = async.execute [%token] -> !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 819 : i64, cycleCost = 164704 : i64, cycleEnd = 165522 : i64} {
      %21 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<416x2048x1x1x!qElemType, #NHWC>) outputs(%3 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
        -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
      async.yield %21 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    }
    %token_8, %bodyResults_9:2 = async.execute -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {
      %21 = VPUIP.SubView %2 [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %22 = VPUIP.SubView %2 [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
            outputs(%21 as %arg5: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, %22 as %arg6: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
            -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [6144, 1024]}(%arg5) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
        VPUIP.SW.Kernel.run {attrs = [432128, 1024]}(%arg6) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      }
      async.yield %results#0, %results#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    }

    %token_10, %bodyResults_11:2 = async.execute [%token_8]
        (%bodyResults_9#0 as %arg1: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, %bodyResults_9#1 as %arg2: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
        -> (!async.value<memref<208x1x1x4xsi32, @DDR>>, !async.value<memref<208x1x1x4xsi32, @DDR>>)
        attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
          cycleBegin = 967 : i64, cycleCost = 57946 : i64, cycleEnd = 58914 : i64} {
      %54 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%5 : memref<208x1x1x4xsi32, @DDR>)
      -> memref<208x1x1x4xsi32, @DDR>
      %55 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>) outputs(%6 : memref<208x1x1x4xsi32, @DDR>)
      -> memref<208x1x1x4xsi32, @DDR>
      async.yield %54, %55 : memref<208x1x1x4xsi32, @DDR>, memref<208x1x1x4xsi32, @DDR>
    }

    %token_12, %bodyResults_13:2 = async.execute [%token_10]
        (%bodyResults_11#0 as %arg1: !async.value<memref<208x1x1x4xsi32, @DDR>>, %bodyResults_11#1 as %arg2: !async.value<memref<208x1x1x4xsi32, @DDR>>)
        -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
        attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 3 : i64,
          cycleBegin = 967 : i64, cycleCost = 57946 : i64, cycleEnd = 58914 : i64} {
      %54 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg1 : memref<208x1x1x4xsi32, @DDR>) outputs(%7 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>)
      -> memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %55 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : memref<208x1x1x4xsi32, @DDR>) outputs(%8 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>)
      -> memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      async.yield %54, %55 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    }

    %token_14, %bodyResults_15 = async.execute [%token_4, %token_6, %token_12] (%bodyResults_5 as %arg2: !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>,
                %bodyResults_13#0 as %arg3: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_13#1 as %arg4: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_7 as %arg5: !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>)
                -> !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 165522 : i64, cycleCost = 1 : i64, cycleEnd = 165523 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg2 : memref<1x1x1x2048xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, [@CMX_NN, 0]>
      %22 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%21 : memref<1x2048x1x1xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>
      %23 = VPUIP.ConcatView inputs(%arg3, %arg4 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>)
        outputs(%2 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
      %24 = VPUIP.NCEClusterTask
                {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
            input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg5 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
            weight_table(%23 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 415], outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %24 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    %result = async.await %bodyResults_15 : !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>

    return %result : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:       -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
    // CHECK-SAME:        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {

    // CHECK:         [[SUBVIEW0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         [[SUBVIEW1:%.+]] = VPUIP.SubView [[IN_BUF]] [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:         [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:       inputs([[SCALE_ARG]] as [[INNER_ARG0:[^:]+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[INNER_ARG2:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, [[SUBVIEW1]] as [[INNER_ARG3:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:       -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [4096, 1024]}([[INNER_ARG2]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [217088, 1024]}([[INNER_ARG3]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:           async.yield [[RESULTS]]#0, [[RESULTS]]#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         }

}

// -----

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<1x1x1x4096xf32, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64}>

!InputDistributedFloat16 = !VPUIP.DistributedBuffer<1x1x1x4096xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64}>

!InputDistributedReshaped = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NCHW, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64}>

!InputDistributedPermuted = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64}>

!PopulateWeightTableDistributed = !VPUIP.DistributedBuffer<512x1x1x4xsi32, { order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>

!WTDistributed = !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>

!WeightsDistributed = !VPUIP.DistributedBuffer<1024x4096x1x1x!qElemType, #NHWC, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>

!ScaleDistributed = !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN,
      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>

!ConvInputDistributed = !VPUIP.DistributedBuffer<1x4096x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments}>

!ConvOutputDistributed = !VPUIP.DistributedBuffer<1x1024x1x1xf16, #NHWC, @CMX_NN,
      {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64,
      alignment = [1, 16, 1, 1]}>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
            attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
}

// CHECK-LABEL: @PatchSWKernelModeSegmented
func.func @PatchSWKernelModeSegmented(%arg0: memref<1x1x4096xf32, @DDR>, %scale: !ScaleDistributed)
           -> !ConvOutputDistributed {

    %0 = VPURT.DeclareBuffer <CMX_NN> <8192> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <0> -> !InputDistributedFloat16
    %2 = VPURT.DeclareBuffer <CMX_NN> <24576> -> !WeightsDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !WTDistributed
    %4 = VPURT.DeclareBuffer <CMX_NN> <1081344> -> !ConvOutputDistributed


    %cst_2 = const.Declare memref<1024x4096x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<1024x4096x1x1xf16>,
              [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]


    %token, %bodyResults:2 = async.execute ->
        (!async.value<!PopulateWeightTableDistributed>, !async.value<!PopulateWeightTableDistributed>)
        attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

      %sub_view0 = VPUIP.SubView %3 [0, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %sub_view1 = VPUIP.SubView %3 [512, 0, 0, 0] [512, 1, 1, 4] {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
            !WTDistributed to !PopulateWeightTableDistributed

      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>, weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
        @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: !ScaleDistributed)
        outputs(%sub_view0 as %arg5: memref<512x1x1x4xsi32, @CMX_NN>, %sub_view1 as %arg6: memref<512x1x1x4xsi32, @CMX_NN>)
            outputStrides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 -> (
            !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
            {
                    VPUIP.SW.Kernel.run {attrs = [8192, 2048]}(%arg5) : memref<512x1x1x4xsi32, @CMX_NN>
                    VPUIP.SW.Kernel.run {attrs = [270336, 2048]}(%arg6) : memref<512x1x1x4xsi32, @CMX_NN>
      }

      async.yield %results#0, %results#1 :
          !PopulateWeightTableDistributed, !PopulateWeightTableDistributed
    }

     %token_3, %bodyResults_4 = async.execute -> !async.value<!InputDistributed>
                                attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 1 : i64,
                                cycleBegin = 0 : i64, cycleCost = 2072 : i64, cycleEnd = 2072 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x4096xf32, @DDR>) -> memref<1x1x1x4096xf32, @DDR>
      %19 = VPUIP.NNDMA {port = 0 : i64} inputs(%18 : memref<1x1x1x4096xf32, @DDR>)
            outputs(%0 : !InputDistributed)
            -> !InputDistributed
      async.yield %19 : !InputDistributed
    }

    %token_7, %bodyResults_8 = async.execute [%token_3] (%bodyResults_4 as %arg2: !async.value<!InputDistributed>)
            -> !async.value<!InputDistributedFloat16>
                attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64,
                cycleBegin = 1637 : i64, cycleCost = 1 : i64, cycleEnd = 1638 : i64} {

      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg2 as %arg3: memref<1x1x1x4096xf32, [@CMX_NN, 0]>)
            outputs(%1 as %arg4: memref<1x1x1x4096xf16, [@CMX_NN, 0]>) on tile 0 -> !InputDistributedFloat16{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x4096xf32, [@CMX_NN, 0]>, memref<1x1x1x4096xf16, [@CMX_NN, 0]>
      }
      async.yield %results : !InputDistributedFloat16
    }
    %token_9, %bodyResults_10 = async.execute -> !async.value<!WeightsDistributed>
                  attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0, 1], "async-deps-index" = 4 : i64,
                  cycleBegin = 2072 : i64, cycleCost = 458697 : i64, cycleEnd = 460769 : i64} {
      %18 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_2 : memref<1024x4096x1x1x!qElemType, #NHWC>) outputs(%2 : !WeightsDistributed)
            -> !WeightsDistributed
      async.yield %18 : !WeightsDistributed
    }
    %token_11, %bodyResults_12 = async.execute [%token, %token_7, %token_9] (%bodyResults_8 as %arg2: !async.value<!InputDistributedFloat16>,
        %bodyResults_10 as %arg3: !async.value<!WeightsDistributed>,
        %bodyResults#0 as %arg4: !async.value<!PopulateWeightTableDistributed>,
        %bodyResults#1 as %arg5: !async.value<!PopulateWeightTableDistributed>)
        -> !async.value<!ConvOutputDistributed>
          attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 460769 : i64, cycleCost = 1 : i64, cycleEnd = 460770 : i64} {
      %18 = VPUIP.GenericReshape inputs(%arg2 : !InputDistributedFloat16)
          -> !InputDistributedReshaped
      %19 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%18 : !InputDistributedReshaped)
          -> !InputDistributedPermuted
      %20 = VPUIP.DistributedCast inputs(%19 : !InputDistributedPermuted)
          -> !ConvInputDistributed
      %21 = VPUIP.ConcatView inputs(%arg4, %arg5 : !PopulateWeightTableDistributed, !PopulateWeightTableDistributed)
        outputs(%3 : !WTDistributed) -> !WTDistributed
      %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1],
                kernel_strides = [1, 1], minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
        input(%20 : !ConvInputDistributed)
        weights(%arg3 : !WeightsDistributed)
        weight_table(%21 : !WTDistributed)
        parent_input(%20 : !ConvInputDistributed)
        parent_output(%4 : !ConvOutputDistributed)
        outputs(%4 : !ConvOutputDistributed)
        -> !ConvOutputDistributed variants : {
              DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 255], outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 511], outStart = [0, 0, 256],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 2 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 767], outStart = [0, 0, 512],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
              DPUTask {cluster_id = 3 : i64, inEnd = [0, 0, 4095], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1023], outStart = [0, 0, 768],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %22 : !ConvOutputDistributed
    }

    %result = async.await %bodyResults_12 : !async.value<!ConvOutputDistributed>

    return %result : !ConvOutputDistributed

    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> <1073152> -> !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN,
    // CHECK-SAME:      {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>

    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:      -> (!async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>>,
    // CHECK-SAME:      !async.value<!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>>)
    // CHECK-SAME:      attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 1 : i64, cycleEnd = 1 : i64} {

    // CHECK:              [[SUBVIEW0:%.+]] = VPUIP.SubView %3 [0, 0, 0, 0] [512, 1, 1, 4]
    // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
    // CHECK-SAME:            !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK-SAME:            to !VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>
    // CHECK:              [[SUBVIEW1:%.+]] = VPUIP.SubView %3 [512, 0, 0, 0] [512, 1, 1, 4]
    // CHECK-SAME{LITERAL}:   {explicit_output_shapes = [[128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4], [128, 1, 1, 4]]} :
    // CHECK-SAME:            !VPUIP.DistributedBuffer<1024x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1]}>
    // CHECK-SAME:            to !VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>
    // CHECK:              [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>,
    // CHECK-SAME:            weightsPtrsPerClusterAttr = [0, 0, 0, 0]}
    // CHECK-SAME:            @VPU.SW::@builtin_PopulateWeightTable inputs(
    // CHECK-SAME:            {{[^:]+}} as {{[^:]+}}: !VPUIP.DistributedBuffer<512x1x1x4xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:            outputs([[SUBVIEW0]] as [[INNER_ARG0:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>,
    // CHECK-SAME:            [[SUBVIEW1]] as [[INNER_ARG1:[^:]+]]: memref<512x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME{LITERAL}:            outputStrides([[4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1], [4, 4, 4, 1]]) on tile 0 ->
    // CHECK-SAME:            (!VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>,
    // CHECK-SAME:            !VPUIP.DistributedBuffer<512x1x1x4xsi32, {order = #NCHW, strides = [4, 4, 4, 1]}, @CMX_NN,
    // CHECK-SAME:              {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64}>){

    // CHECK:                VPUIP.SW.Kernel.run {attrs = [24576, 2048]}([[INNER_ARG0]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:                VPUIP.SW.Kernel.run {attrs = [1073152, 2048]}([[INNER_ARG1]]) : memref<512x1x1x4xsi32, @CMX_NN>
    // CHECK:              }
    // CHECK:              async.yield [[RESULTS]]#0, [[RESULTS]]#1
    // CHECK:       }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 0.0057189941406250002>

module @VPU.SW {
    func.func private @builtin_PopulateWeightTable(memref<*xsi32, @CMX_NN>, i64, i64)
            attributes {VPU.kernel_code = "populate_weight_table.cpp",
                        VPU.kernel_entry = "populate_weight_table", VPU.task_type = @COMPUTE}
    func.func private @builtin_Convert(memref<*xf32, [@CMX_NN, 0]>, memref<*xf16, [@CMX_NN, 0]>)
            attributes {VPU.kernel_code = "convert.cpp", VPU.kernel_entry = "convert", VPU.task_type = @COMPUTE}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
// CHECK-LABEL: @DoNotPatchSWKernelWhenZeroOffset
// CHECK-SAME:      , [[SCALE_ARG:%.+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>
func.func @DoNotPatchSWKernelWhenZeroOffset(%arg0: memref<1x1x2048xf32, @DDR>, %scale: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
           -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <4096> -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <856064> -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>

    %cst = const.Declare memref<416x2048x1x1x!qElemType, #NHWC> = dense<1.0> : tensor<416x2048x1x1xf16>,
        [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %token, %bodyResults = async.execute -> !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 0 : i64, cycleBegin = 0 : i64, cycleCost = 819 : i64, cycleEnd = 819 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1x2048xf32, @DDR>) -> memref<1x1x1x2048xf32, @DDR>
      %22 = VPUIP.NNDMA {port = 0 : i64} inputs(%21 : memref<1x1x1x2048xf32, @DDR>) outputs(%0 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>) -> memref<1x1x1x2048xf32, [@CMX_NN, 0]>
      async.yield %22 : memref<1x1x1x2048xf32, [@CMX_NN, 0]>
    }
    %token_4, %bodyResults_5 = async.execute [%token] (%bodyResults as %arg2: !async.value<memref<1x1x1x2048xf32, [@CMX_NN, 0]>>) -> !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 1 : i64, cycleBegin = 819 : i64, cycleCost = 1 : i64, cycleEnd = 820 : i64} {
      %results = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_Convert inputs(%arg2 as %arg3: memref<1x1x1x2048xf32, [@CMX_NN, 0]>)
            outputs(%1 as %arg4: memref<1x1x1x2048xf16, [@CMX_NN, 0]>) on tile 0 -> memref<1x1x1x2048xf16, [@CMX_NN, 0]>{
        VPUIP.SW.Kernel.run(%arg3, %arg4) : memref<1x1x1x2048xf32, [@CMX_NN, 0]>, memref<1x1x1x2048xf16, [@CMX_NN, 0]>
      }
      async.yield %results : memref<1x1x1x2048xf16, [@CMX_NN, 0]>
    }
    %token_6, %bodyResults_7 = async.execute [%token] -> !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>
            attributes {VPUIP.executor = @DMA_NN, VPUIP.executorIdx = [0], "async-deps-index" = 2 : i64, cycleBegin = 819 : i64, cycleCost = 164704 : i64, cycleEnd = 165522 : i64} {
      %21 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<416x2048x1x1x!qElemType, #NHWC>) outputs(%3 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
        -> memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
      async.yield %21 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>
    }

    %token_8, %bodyResults_9:2 = async.execute -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
            attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {
      %21 = VPUIP.SubView %2 [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %22 = VPUIP.SubView %2 [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      %results:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable inputs(%scale as %arg3: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
            outputs(%21 as %arg5: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, %22 as %arg6: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
            -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
        VPUIP.SW.Kernel.run {attrs = [6144, 1024]}(%arg5) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
        VPUIP.SW.Kernel.run {attrs = [432128, 1024]}(%arg6) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
      }
      async.yield %results#0, %results#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    }

    %token_12, %bodyResults_13 = async.execute [%token_4, %token_6, %token_8] (%bodyResults_5 as %arg2: !async.value<memref<1x1x1x2048xf16, [@CMX_NN, 0]>>,
                %bodyResults_9#0 as %arg3: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_9#1 as %arg4: !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>,
                %bodyResults_7 as %arg5: !async.value<memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>>)
                -> !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>
                attributes {VPUIP.executor = @DPU, "async-deps-index" = 5 : i64, cycleBegin = 165522 : i64, cycleCost = 1 : i64, cycleEnd = 165523 : i64} {
      %21 = VPUIP.GenericReshape inputs(%arg2 : memref<1x1x1x2048xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, [@CMX_NN, 0]>
      %22 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%21 : memref<1x2048x1x1xf16, [@CMX_NN, 0]>) -> memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>
      %23 = VPUIP.ConcatView inputs(%arg3, %arg4 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>)
        outputs(%2 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>) -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>
      %24 = VPUIP.NCEClusterTask
                {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
                minimumHardwareExecutionCost = 4294967295 : i64, populateWeightTable = true, task_type = #VPUIP.nce_task_type<CONV>}
            input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%arg5 : memref<416x2048x1x1x!qElemType, #NHWC, [@CMX_NN, 0]>)
            weight_table(%23 : memref<416x1x1x4xsi32, [@CMX_NN, 0]>)
            parent_input(%22 : memref<1x2048x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%4 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            -> memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 415], outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
      async.yield %24 : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>
    }

    %result = async.await %bodyResults_13 : !async.value<memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>>

    return %result : memref<1x416x1x1xf16, #NHWC, [@CMX_NN, 0]>

    // CHECK:       [[IN_BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <864256> -> memref<416x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:       [[TOKEN:%.+]], [[BODY_RESULT:%.+]]:2 = async.execute
    // CHECK-SAME:      -> (!async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>, !async.value<memref<208x1x1x4xsi32, [@CMX_NN, 0]>>)
    // CHECK-SAME:       attributes {VPUIP.executor = @SHAVE_ACT, "async-deps-index" = 3 : i64, cycleBegin = 820 : i64, cycleCost = 1 : i64, cycleEnd = 821 : i64} {

    // CHECK:         [[SUBVIEW0:%.+]] = VPUIP.SubView [[IN_BUF]] [0, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         [[SUBVIEW1:%.+]] = VPUIP.SubView [[IN_BUF]] [208, 0, 0, 0] [208, 1, 1, 4] : memref<416x1x1x4xsi32, [@CMX_NN, 0]> to memref<208x1x1x4xsi32, [@CMX_NN, 0]>

    // CHECK:         [[RESULTS:%.+]]:2 = VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 2, 0, 0>} @VPU.SW::@builtin_PopulateWeightTable
    // CHECK-SAME:       inputs([[SCALE_ARG]] as [[INNER_ARG0:[^:]+]]: memref<416x1x1x4xf16, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[SUBVIEW0]] as [[INNER_ARG2:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>, [[SUBVIEW1]] as [[INNER_ARG3:[^:]+]]: memref<208x1x1x4xsi32, [@CMX_NN, 0]>) on tile 0
    // CHECK-SAME:       -> (memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>){
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [6144, 1024]}([[INNER_ARG2]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:               VPUIP.SW.Kernel.run {attrs = [432128, 1024]}([[INNER_ARG3]]) : memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:           }
    // CHECK:           async.yield [[RESULTS]]#0, [[RESULTS]]#1 : memref<208x1x1x4xsi32, [@CMX_NN, 0]>, memref<208x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:         }
}
